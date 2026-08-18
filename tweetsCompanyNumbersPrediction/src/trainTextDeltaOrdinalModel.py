"""Rolling future evaluation of a text-delta ordinal multi-task quarter model."""

import argparse
import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_absolute_error
from torch.utils.data import DataLoader

from classifier.QuarterAlignedDataset import reporting_quarters
from classifier.TextDeltaOrdinalModel import TextDeltaOrdinalModel, ordinal_targets
from classifier.TextDeltaQuarterDataset import (
    TextDeltaQuarterDataset,
    TextDeltaQuarterRecord,
    aggregate_quarter_text_features,
    build_text_delta_records,
    shuffle_text_within_company,
)
from trainQuarterSequenceModel import COMPANIES, baseline_metrics


COMPANY_EXPERIMENTS = {
    "amazon": "amazon-revenue-4class",
    "apple": "apple-eps",
    "tesla": "tesla-sales",
}
HEADS = ("finance", "text", "fusion")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(COMPANIES))
    parser.add_argument("--test-years", default="2017,2018,2019")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--max-relevant-per-bin", type=int, default=512)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sentence-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--text-weight", type=float, default=0.4)
    parser.add_argument("--text-epochs", type=int, default=60)
    parser.add_argument("--fusion-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cache", default="../../output/text_delta_quarter_cache.npz")
    parser.add_argument("--output", default="../../output/text_delta_ordinal_results.json")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cache_signature(args, companies, maximum_year):
    return {
        "version": 1,
        "companies": companies,
        "maximum_year": maximum_year,
        "bins": args.bins,
        "max_relevant_per_bin": args.max_relevant_per_bin,
        "lookback": args.lookback,
        "sentence_model": args.sentence_model,
        "seed": args.seed,
        "feature_design": "mean_embedding+12_stats,current+qoq+yoy+flags",
    }


def load_cached_records(path, signature):
    if not path or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as cache:
        if json.loads(str(cache["signature"].item())) != signature:
            return None
        records = []
        for index in range(int(cache["record_count"].item())):
            records.append(TextDeltaQuarterRecord(
                company=str(cache["company_%d" % index].item()),
                company_index=int(cache["company_index_%d" % index].item()),
                quarter=str(cache["quarter_%d" % index].item()),
                label=int(cache["label_%d" % index].item()),
                percent_change=float(cache["percent_change_%d" % index].item()),
                text_sequence=cache["text_%d" % index],
                financial_sequence=cache["financial_%d" % index],
            ))
    print("Loaded text-delta cache", path)
    return records


def save_cached_records(path, signature, records):
    if not path:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    values = {
        "signature": np.asarray(json.dumps(signature, sort_keys=True)),
        "record_count": np.asarray(len(records)),
    }
    for index, record in enumerate(records):
        values["company_%d" % index] = np.asarray(record.company)
        values["company_index_%d" % index] = np.asarray(record.company_index)
        values["quarter_%d" % index] = np.asarray(record.quarter)
        values["label_%d" % index] = np.asarray(record.label)
        values["percent_change_%d" % index] = np.asarray(record.percent_change)
        values["text_%d" % index] = record.text_sequence
        values["financial_%d" % index] = record.financial_sequence
    np.savez_compressed(path, **values)
    print("Wrote text-delta cache", path)


def build_records(companies, maximum_year, args, device):
    signature = cache_signature(args, companies, maximum_year)
    cache_path = os.path.abspath(args.cache) if args.cache else None
    cached = load_cached_records(cache_path, signature)
    if cached is not None:
        return cached

    print("Loading frozen sentence encoder", args.sentence_model)
    sentence_model = SentenceTransformer(args.sentence_model, device=str(device))
    records = []
    for company_index, company in enumerate(companies):
        prediction_path = COMPANIES[company]
        financial_frame = pd.read_csv(prediction_path.getFinancialNumbersPath())
        tweets = pd.read_csv(
            prediction_path.getDataframePath(), usecols=["post_date", "body", "class"])
        tweets["body"] = tweets["body"].fillna("").astype(str)
        tweets["reporting_quarter"] = reporting_quarters(tweets["post_date"])
        quarters = sorted({
            quarter for quarter in tweets["reporting_quarter"].unique()
            if 2015 <= int(quarter[:4]) <= maximum_year
        })
        print("\n%s: %d tweets, %d target quarters" % (company, len(tweets), len(quarters)))
        base_features = aggregate_quarter_text_features(
            tweets, quarters, sentence_model, COMPANY_EXPERIMENTS[company],
            bins=args.bins, max_relevant_per_bin=args.max_relevant_per_bin,
            batch_size=args.sentence_batch_size, seed=args.seed + company_index * 1000)
        company_records = build_text_delta_records(
            company, company_index, tweets, financial_frame, base_features,
            lookback=args.lookback)
        records.extend(company_records)
    del sentence_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    save_cached_records(cache_path, signature, records)
    return records


def split_records(records, validation_year, test_year):
    train = [record for record in records if int(record.quarter[:4]) < validation_year]
    validation = [record for record in records if int(record.quarter[:4]) == validation_year]
    test = [record for record in records if int(record.quarter[:4]) == test_year]
    if not train or not validation or not test:
        raise ValueError("Each rolling fold requires train, validation and test records")
    return train, validation, test


def make_loader(records, args, shuffle):
    return DataLoader(
        TextDeltaQuarterDataset(records), batch_size=args.batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=torch.cuda.is_available())


def target_statistics(records, num_companies, device):
    means, standard_deviations = [], []
    for company_index in range(num_companies):
        values = np.asarray([
            record.percent_change for record in records
            if record.company_index == company_index
        ], dtype=np.float32)
        means.append(float(values.mean()))
        standard_deviations.append(max(float(values.std()), 10.0))
    return (
        torch.tensor(means, dtype=torch.float32, device=device),
        torch.tensor(standard_deviations, dtype=torch.float32, device=device),
    )


def create_model(records, args, device):
    return TextDeltaOrdinalModel(
        text_feature_size=records[0].text_sequence.shape[-1],
        financial_feature_size=records[0].financial_sequence.shape[-1],
        num_companies=len(set(record.company for record in records)),
        hidden_size=args.hidden_size,
        text_weight=args.text_weight,
    ).to(device)


def create_losses(records, device):
    labels = np.asarray([record.label for record in records], dtype=int)
    counts = np.bincount(labels, minlength=4).astype(np.float32)
    class_weights = len(labels) / np.maximum(counts, 1.0)
    class_weights /= class_weights.mean()
    ordinal = np.asarray([[label > threshold for threshold in range(3)] for label in labels])
    positives = ordinal.sum(axis=0).astype(np.float32)
    negatives = len(labels) - positives
    positive_weights = np.clip(negatives / np.maximum(positives, 1.0), 0.25, 4.0)
    return {
        "class": torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=device), label_smoothing=0.02),
        "ordinal": torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weights, device=device)),
        "regression": torch.nn.SmoothL1Loss(),
    }


def move_batch(batch, device):
    text, financial, company, quarter, labels, changes, indexes = batch
    return (
        text.to(device), financial.to(device), company.to(device), quarter.to(device),
        labels.to(device), changes.to(device), indexes,
    )


def loss_components(outputs, labels, changes, company, losses, statistics):
    means, standard_deviations = statistics
    regression_targets = (changes - means[company]) / standard_deviations[company]
    return {
        "fusion": losses["class"](outputs["fusion"], labels),
        "text": losses["class"](outputs["text"], labels),
        "finance": losses["class"](outputs["finance"], labels),
        "ordinal": losses["ordinal"](outputs["ordinal"], ordinal_targets(labels)),
        "regression": losses["regression"](outputs["regression"], regression_targets),
    }


def combined_loss(components, stage):
    if stage == "text":
        return components["text"] + 0.65 * components["ordinal"] + 0.35 * components["regression"]
    return (
        components["fusion"] + 0.70 * components["text"]
        + 0.30 * components["ordinal"] + 0.25 * components["regression"]
        + 0.30 * components["finance"]
    )


def train_epoch(model, loader, optimizer, losses, statistics, stage, device):
    model.train()
    values = []
    for batch in loader:
        text, financial, company, quarter, labels, changes, _ = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.all_outputs(text, financial, company, quarter)
        loss = combined_loss(
            loss_components(outputs, labels, changes, company, losses, statistics), stage)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        values.append(float(loss.detach().cpu()))
    return float(np.mean(values))


@torch.no_grad()
def predict(model, records, args, losses, statistics, stage, device):
    model.eval()
    probabilities = {head: [] for head in HEADS}
    regression_values, validation_losses = [], []
    for batch in make_loader(records, args, shuffle=False):
        text, financial, company, quarter, labels, changes, _ = move_batch(batch, device)
        outputs = model.all_outputs(text, financial, company, quarter)
        components = loss_components(
            outputs, labels, changes, company, losses, statistics)
        validation_losses.append(float(combined_loss(components, stage).cpu()))
        for head in HEADS:
            probabilities[head].append(torch.softmax(outputs[head], dim=1).cpu().numpy())
        means, standard_deviations = statistics
        raw_regression = outputs["regression"] * standard_deviations[company] + means[company]
        regression_values.extend(raw_regression.cpu().numpy().tolist())
    return (
        float(np.mean(validation_losses)),
        {head: np.concatenate(values, axis=0) for head, values in probabilities.items()},
        np.asarray(regression_values),
    )


def metric_summary(records, probabilities):
    labels = np.asarray([record.label for record in records])
    predictions = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "true": labels.tolist(),
        "predicted": predictions.tolist(),
        "companies": [record.company for record in records],
        "quarters": [record.quarter for record in records],
    }
    result["per_company"] = {}
    for company in sorted(set(record.company for record in records)):
        indexes = [index for index, record in enumerate(records) if record.company == company]
        result["per_company"][company] = {
            "accuracy": float(accuracy_score(labels[indexes], predictions[indexes])),
            "mcc": float(matthews_corrcoef(labels[indexes], predictions[indexes])),
        }
    return result


def fit_stage(model, train_records, validation_records, args, device, statistics,
              losses, stage, maximum_epochs, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=3e-3)
    loader = make_loader(train_records, args, shuffle=True)
    best_loss, best_epoch, stale, best_state = float("inf"), 1, 0, None
    for epoch in range(1, maximum_epochs + 1):
        train_loss = train_epoch(
            model, loader, optimizer, losses, statistics, stage, device)
        validation_loss, probabilities, _ = predict(
            model, validation_records, args, losses, statistics, stage, device)
        metrics = metric_summary(validation_records, probabilities["text" if stage == "text" else "fusion"])
        print("  %s %02d train %.4f val %.4f acc %.3f MCC %.3f"
              % (stage, epoch, train_loss, validation_loss,
                 metrics["accuracy"], metrics["mcc"]))
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return best_epoch, best_loss


def select_epochs(train_records, validation_records, args, device, seed):
    seed_everything(seed)
    model = create_model(train_records + validation_records, args, device)
    statistics = target_statistics(
        train_records, len(set(record.company for record in train_records)), device)
    losses = create_losses(train_records, device)
    text_epoch, text_loss = fit_stage(
        model, train_records, validation_records, args, device, statistics,
        losses, "text", args.text_epochs, 7e-4)
    fusion_epoch, fusion_loss = fit_stage(
        model, train_records, validation_records, args, device, statistics,
        losses, "fusion", args.fusion_epochs, 4e-4)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return text_epoch, fusion_epoch, text_loss, fusion_loss


def refit_and_test(train_records, validation_records, test_records, text_epochs,
                   fusion_epochs, args, device, seed):
    seed_everything(seed)
    combined = train_records + validation_records
    model = create_model(combined, args, device)
    statistics = target_statistics(
        combined, len(set(record.company for record in combined)), device)
    losses = create_losses(combined, device)
    loader = make_loader(combined, args, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-3)
    for epoch in range(1, text_epochs + 1):
        loss = train_epoch(model, loader, optimizer, losses, statistics, "text", device)
        print("  text refit %02d/%02d %.4f" % (epoch, text_epochs, loss))
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=3e-3)
    for epoch in range(1, fusion_epochs + 1):
        loss = train_epoch(model, loader, optimizer, losses, statistics, "fusion", device)
        print("  fusion refit %02d/%02d %.4f" % (epoch, fusion_epochs, loss))
    _, probabilities, regression = predict(
        model, test_records, args, losses, statistics, "fusion", device)
    shuffled_records = shuffle_text_within_company(test_records, seed + 17)
    _, shuffled_probabilities, _ = predict(
        model, shuffled_records, args, losses, statistics, "fusion", device)
    regression_mae = float(mean_absolute_error(
        [record.percent_change for record in test_records], regression))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, shuffled_probabilities["fusion"], regression_mae


def aggregate_baselines(fold_baselines):
    results = {}
    for name in fold_baselines[0]:
        true = sum((fold[name]["true"] for fold in fold_baselines), [])
        predicted = sum((fold[name]["predicted"] for fold in fold_baselines), [])
        results[name] = {
            "accuracy": float(accuracy_score(true, predicted)),
            "mcc": float(matthews_corrcoef(true, predicted)),
        }
    return results


def main():
    args = parse_arguments()
    companies = [value.strip() for value in args.companies.split(",") if value.strip()]
    test_years = [int(value.strip()) for value in args.test_years.split(",") if value.strip()]
    if any(company not in COMPANIES for company in companies):
        raise ValueError("Unknown company in %s" % companies)
    if min(test_years) < 2017:
        raise ValueError("Future folds require an earlier train and validation year")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)
    print("Fixed normalized text weight:", args.text_weight)
    records = build_records(companies, max(test_years), args, device)
    print("Built %d independent company-quarter records with text shape %s"
          % (len(records), records[0].text_sequence.shape))

    probabilities_by_head = {
        head: [[] for _ in range(args.runs)] for head in HEADS}
    shuffled_by_run = [[] for _ in range(args.runs)]
    all_test_records, folds, fold_baselines = [], [], []
    for test_year in test_years:
        validation_year = test_year - 1
        train_records, validation_records, test_records = split_records(
            records, validation_year, test_year)
        print("\n=== test %d: train < %d, validation %d, test %d ==="
              % (test_year, validation_year, validation_year, test_year))
        print("Independent records:", len(train_records), len(validation_records), len(test_records))
        fold_baseline = baseline_metrics(train_records + validation_records, test_records)
        fold_baselines.append(fold_baseline)
        fold = {"test_year": test_year, "runs": [], "baselines": fold_baseline}
        all_test_records.extend(test_records)
        for run in range(args.runs):
            seed = args.seed + test_year + run * 100
            print("\nRun %d/%d seed %d" % (run + 1, args.runs, seed))
            text_epochs, fusion_epochs, text_loss, fusion_loss = select_epochs(
                train_records, validation_records, args, device, seed)
            probabilities, shuffled, regression_mae = refit_and_test(
                train_records, validation_records, test_records, text_epochs,
                fusion_epochs, args, device, seed)
            run_result = {
                head: metric_summary(test_records, probabilities[head]) for head in HEADS}
            run_result["fusion_shuffled_test"] = metric_summary(test_records, shuffled)
            run_result.update({
                "seed": seed,
                "selected_text_epochs": text_epochs,
                "selected_fusion_epochs": fusion_epochs,
                "validation_text_loss": text_loss,
                "validation_fusion_loss": fusion_loss,
                "percent_change_mae": regression_mae,
            })
            fold["runs"].append(run_result)
            for head in HEADS:
                probabilities_by_head[head][run].append(probabilities[head])
            shuffled_by_run[run].append(shuffled)
            print("  TEST text %.3f/%.3f fusion %.3f/%.3f shuffled %.3f/%.3f"
                  % (run_result["text"]["accuracy"], run_result["text"]["mcc"],
                     run_result["fusion"]["accuracy"], run_result["fusion"]["mcc"],
                     run_result["fusion_shuffled_test"]["accuracy"],
                     run_result["fusion_shuffled_test"]["mcc"]))
        folds.append(fold)

    models = {}
    for head in HEADS:
        run_summaries, combined = [], []
        for run in range(args.runs):
            values = np.concatenate(probabilities_by_head[head][run], axis=0)
            combined.append(values)
            run_summaries.append(metric_summary(all_test_records, values))
        models[head] = {
            "accuracy_mean": float(np.mean([value["accuracy"] for value in run_summaries])),
            "accuracy_std": float(np.std([value["accuracy"] for value in run_summaries])),
            "mcc_mean": float(np.mean([value["mcc"] for value in run_summaries])),
            "mcc_std": float(np.std([value["mcc"] for value in run_summaries])),
            "ensemble": metric_summary(all_test_records, np.mean(combined, axis=0)),
            "runs": run_summaries,
        }
    shuffled_summaries, shuffled_combined = [], []
    for run in range(args.runs):
        values = np.concatenate(shuffled_by_run[run], axis=0)
        shuffled_combined.append(values)
        shuffled_summaries.append(metric_summary(all_test_records, values))
    models["fusion_shuffled_test"] = {
        "accuracy_mean": float(np.mean([value["accuracy"] for value in shuffled_summaries])),
        "accuracy_std": float(np.std([value["accuracy"] for value in shuffled_summaries])),
        "mcc_mean": float(np.mean([value["mcc"] for value in shuffled_summaries])),
        "mcc_std": float(np.std([value["mcc"] for value in shuffled_summaries])),
        "ensemble": metric_summary(all_test_records, np.mean(shuffled_combined, axis=0)),
        "runs": shuffled_summaries,
    }

    for name, values in models.items():
        print("%s mean accuracy %.4f (+/- %.4f), MCC %.4f (+/- %.4f); ensemble %.4f/%.4f"
              % (name, values["accuracy_mean"], values["accuracy_std"],
                 values["mcc_mean"], values["mcc_std"],
                 values["ensemble"]["accuracy"], values["ensemble"]["mcc"]))

    results = {
        "protocol": "rolling-origin future-only; train < validation < test",
        "target": "four-class current-quarter percent_change",
        "auxiliary_target": "continuous percent_change from the same quarterly CSV",
        "data_scope": "local tweets and quarterly financial CSVs only; frozen sentence encoder",
        "independent_test_company_quarters": len(all_test_records),
        "text_weight": args.text_weight,
        "text_features": "finance-relevant bin means and lexical statistics; current, QoQ and YoY deltas",
        "baselines": aggregate_baselines(fold_baselines),
        "models": models,
        "folds": folds,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
