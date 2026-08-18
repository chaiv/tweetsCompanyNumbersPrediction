"""Pooled company-quarter LSTM ablation using only local tweets and financial CSVs.

The target is always the four-class bucket of the current quarter's percentage change.  Current
quarter text is allowed for nowcasting, whereas every financial input is strictly lagged (t-1 to
t-4).  Metrics are computed after averaging bag predictions to one decision per company-quarter.
"""

import argparse
import json
import os
import random
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import ConcatDataset, DataLoader

from PredictionModelPath import (
    AMAZON_REVENUE_10_LSTM_MULTI_CLASS,
    APPLE__EPS_10_LSTM_MULTI_CLASS,
    TESLA_CAR_SALES_10_LSTM_MULTI_CLASS,
)
from classifier.QuarterAlignedDataset import reporting_quarters
from classifier.QuarterSequenceDataset import (
    FINANCIAL_FEATURE_NAMES,
    QuarterSequenceDataset,
    QuarterSequenceRecord,
    lagged_financial_sequence,
    prepare_financial_quarters,
    select_text_bags,
)
from classifier.QuarterSequenceModel import ARCHITECTURE_VIEWS, QuarterSequenceClassifier


COMPANIES = {
    "amazon": AMAZON_REVENUE_10_LSTM_MULTI_CLASS,
    "apple": APPLE__EPS_10_LSTM_MULTI_CLASS,
    "tesla": TESLA_CAR_SALES_10_LSTM_MULTI_CLASS,
}

EXPERIMENT_ARCHITECTURES = dict(ARCHITECTURE_VIEWS)
EXPERIMENT_ARCHITECTURES["fusion-shuffled-text"] = ARCHITECTURE_VIEWS["fusion"]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(COMPANIES))
    parser.add_argument("--architectures", default=",".join(EXPERIMENT_ARCHITECTURES))
    parser.add_argument("--validation-year", type=int, default=2018)
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--tweets-per-bin", type=int, default=8)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sentence-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cache", default="../../output/quarter_sequence_text_cache.npz")
    parser.add_argument("--output", default="../../output/quarter_sequence_results.json")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cache_signature(args, company_names):
    return {
        "version": 2,
        "companies": company_names,
        "validation_year": args.validation_year,
        "test_year": args.test_year,
        "bins": args.bins,
        "tweets_per_bin": args.tweets_per_bin,
        "variants": args.variants,
        "lookback": args.lookback,
        "sentence_model": args.sentence_model,
        "seed": args.seed,
    }


def load_cached_records(path, signature):
    if not path or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as cache:
        cached_signature = json.loads(str(cache["signature"].item()))
        if cached_signature != signature:
            return None
        records = []
        count = int(cache["record_count"].item())
        for index in range(count):
            records.append(QuarterSequenceRecord(
                company=str(cache["company_%d" % index].item()),
                company_index=int(cache["company_index_%d" % index].item()),
                quarter=str(cache["quarter_%d" % index].item()),
                label=int(cache["label_%d" % index].item()),
                text_sequences=cache["text_%d" % index],
                financial_sequence=cache["financial_%d" % index],
            ))
    print("Loaded cached frozen text embeddings from", path)
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
        values["text_%d" % index] = record.text_sequences
        values["financial_%d" % index] = record.financial_sequence
    np.savez_compressed(path, **values)
    print("Wrote frozen text embedding cache", path)


def encode_company_records(company, company_index, prediction_path, sentence_model, args):
    financial = prepare_financial_quarters(pd.read_csv(prediction_path.getFinancialNumbersPath()))
    allowed_years = range(2015, args.test_year + 1)
    quarters = financial[
        financial["quarter"].str[:4].astype(int).isin(allowed_years)
        & financial["label"].notna()
    ]["quarter"].tolist()

    tweets = pd.read_csv(
        prediction_path.getDataframePath(), usecols=["post_date", "body", "class"])
    tweets["body"] = tweets["body"].fillna("")
    tweets["reporting_quarter"] = reporting_quarters(tweets["post_date"])
    print("%s: selecting text from %d tweets across %d quarters"
          % (company, len(tweets), len(quarters)))

    selected_bags = []
    labels = []
    for offset, quarter in enumerate(quarters):
        tweet_labels = tweets.loc[tweets["reporting_quarter"] == quarter, "class"].dropna().unique()
        if len(tweet_labels) != 1:
            raise ValueError("%s %s has tweet labels %s" % (company, quarter, tweet_labels))
        financial_label = int(financial.loc[financial["quarter"] == quarter, "label"].iloc[0])
        if int(tweet_labels[0]) != financial_label:
            raise ValueError("%s %s: tweet label %s != financial label %s"
                             % (company, quarter, tweet_labels[0], financial_label))
        labels.append(financial_label)
        selected_bags.append(select_text_bags(
            tweets,
            quarter,
            bins=args.bins,
            tweets_per_bin=args.tweets_per_bin,
            variants=args.variants,
            seed=args.seed + company_index * 1000 + offset,
        ))

    selected = np.asarray(selected_bags, dtype=object)
    flattened_texts = selected.reshape(-1).tolist()
    print("%s: encoding %d selected tweets with frozen MiniLM" % (company, len(flattened_texts)))
    embeddings = sentence_model.encode(
        flattened_texts,
        batch_size=args.sentence_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embedding_size = embeddings.shape[1]
    text_sequences = embeddings.reshape(
        len(quarters), args.variants, args.bins, args.tweets_per_bin, embedding_size).mean(axis=3)

    records = []
    for index, quarter in enumerate(quarters):
        records.append(QuarterSequenceRecord(
            company=company,
            company_index=company_index,
            quarter=quarter,
            label=labels[index],
            text_sequences=text_sequences[index],
            financial_sequence=lagged_financial_sequence(
                financial, quarter, lookback=args.lookback),
        ))
    return records


def build_records(company_names, args, device):
    signature = cache_signature(args, company_names)
    cache_path = os.path.abspath(args.cache) if args.cache else None
    cached = load_cached_records(cache_path, signature)
    if cached is not None:
        return cached

    print("Loading frozen sentence encoder", args.sentence_model)
    sentence_model = SentenceTransformer(args.sentence_model, device=str(device))
    records = []
    for company_index, company in enumerate(company_names):
        records.extend(encode_company_records(
            company, company_index, COMPANIES[company], sentence_model, args))
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
        raise ValueError("Train, validation and test must each contain at least one quarter")
    return train, validation, test


def shuffle_text_within_company(records, seed):
    """Negative control: break quarter/text alignment while retaining company identity."""
    shuffled = list(records)
    random_state = np.random.RandomState(seed)
    for company in sorted(set(record.company for record in records)):
        indexes = [index for index, record in enumerate(records) if record.company == company]
        if len(indexes) < 2:
            continue
        shift = int(random_state.randint(1, len(indexes)))
        source_indexes = indexes[shift:] + indexes[:shift]
        for target_index, source_index in zip(indexes, source_indexes):
            shuffled[target_index] = replace(
                records[target_index], text_sequences=records[source_index].text_sequences)
    return shuffled


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def create_model(architecture, records, args, device):
    model_architecture = "fusion" if architecture == "fusion-shuffled-text" else architecture
    return QuarterSequenceClassifier(
        sentence_embedding_size=records[0].text_sequences.shape[-1],
        financial_feature_size=records[0].financial_sequence.shape[-1],
        num_companies=len(set(record.company for record in records)),
        num_classes=4,
        architecture=model_architecture,
        hidden_size=args.hidden_size,
    ).to(device)


def class_weights(records, device):
    counts = np.bincount([record.label for record in records], minlength=4).astype(np.float32)
    weights = len(records) / np.maximum(counts, 1.0)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    losses = []
    for text, financial, company, quarter, labels, _ in loader:
        text, financial = text.to(device), financial.to(device)
        company, quarter, labels = company.to(device), quarter.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(text, financial, company, quarter)
        loss = loss_function(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict_variants(model, dataset, batch_size, loss_function, device):
    model.eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    probabilities = [[] for _ in dataset.records]
    losses = []
    for text, financial, company, quarter, labels, record_indexes in loader:
        text, financial = text.to(device), financial.to(device)
        company, quarter, labels = company.to(device), quarter.to(device), labels.to(device)
        logits = model(text, financial, company, quarter)
        losses.append(float(loss_function(logits, labels).cpu()))
        batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        for probability, record_index in zip(batch_probabilities, record_indexes.tolist()):
            probabilities[record_index].append(probability)
    aggregated = np.asarray([np.mean(values, axis=0) for values in probabilities])
    return float(np.mean(losses)), aggregated


def metric_summary(records, probabilities):
    labels = np.asarray([record.label for record in records])
    predictions = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "true": labels.tolist(),
        "predicted": predictions.tolist(),
        "quarters": [record.quarter for record in records],
        "companies": [record.company for record in records],
    }
    per_company = {}
    for company in sorted(set(record.company for record in records)):
        indexes = [index for index, record in enumerate(records) if record.company == company]
        per_company[company] = {
            "accuracy": float(accuracy_score(labels[indexes], predictions[indexes])),
            "mcc": float(matthews_corrcoef(labels[indexes], predictions[indexes])),
            "true": labels[indexes].tolist(),
            "predicted": predictions[indexes].tolist(),
        }
    result["per_company"] = per_company
    return result


def fit_with_validation(architecture, train_records, validation_records, args, device, seed):
    seed_everything(seed)
    train_dataset = QuarterSequenceDataset(train_records)
    validation_dataset = QuarterSequenceDataset(validation_records)
    model = create_model(architecture, train_records + validation_records, args, device)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights(train_records, device), label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-3)
    loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    best_loss, best_epoch, stale = float("inf"), 1, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, loss_function, device)
        validation_loss, probabilities = predict_variants(
            model, validation_dataset, args.batch_size, loss_function, device)
        metrics = metric_summary(validation_records, probabilities)
        print("  epoch %02d train_loss %.4f val_loss %.4f val_accuracy %.4f val_mcc %.4f"
              % (epoch, train_loss, validation_loss, metrics["accuracy"], metrics["mcc"]))
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, best_loss


def refit_and_test(architecture, train_records, validation_records, test_records,
                   epochs, args, device, seed):
    seed_everything(seed)
    combined_records = train_records + validation_records
    combined_dataset = QuarterSequenceDataset(combined_records)
    test_dataset = QuarterSequenceDataset(test_records)
    model = create_model(architecture, combined_records, args, device)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights(combined_records, device), label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-3)
    loader = make_loader(combined_dataset, args.batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, loss_function, device)
        print("  refit epoch %02d/%02d loss %.4f" % (epoch, epochs, loss))
    test_loss, probabilities = predict_variants(
        model, test_dataset, args.batch_size, loss_function, device)
    metrics = metric_summary(test_records, probabilities)
    metrics["test_loss"] = test_loss
    metrics["probabilities"] = probabilities.tolist()
    metrics["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, probabilities


def baseline_metrics(train_records, test_records):
    def summarize(predictions):
        labels = [record.label for record in test_records]
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "mcc": float(matthews_corrcoef(labels, predictions)),
            "true": labels,
            "predicted": [int(value) for value in predictions],
        }

    company_majorities = {}
    seasonal = {}
    for company in sorted(set(record.company for record in train_records)):
        company_records = [record for record in train_records if record.company == company]
        labels = [record.label for record in company_records]
        company_majorities[company] = int(np.bincount(labels, minlength=4).argmax())
        for calendar_quarter in range(1, 5):
            seasonal_labels = [
                record.label for record in company_records
                if int(record.quarter[-1]) == calendar_quarter]
            seasonal[(company, calendar_quarter)] = int(
                np.bincount(seasonal_labels, minlength=4).argmax())

    majority_predictions = [company_majorities[record.company] for record in test_records]
    seasonal_predictions = [
        seasonal[(record.company, int(record.quarter[-1]))] for record in test_records]
    previous_predictions = []
    year_ago_predictions = []
    for record in test_records:
        previous_change = float(record.financial_sequence[-1, 1]) * 100.0
        year_ago_change = float(record.financial_sequence[0, 1]) * 100.0
        from classifier.QuarterSequenceDataset import percent_change_class
        previous_predictions.append(percent_change_class(previous_change))
        year_ago_predictions.append(percent_change_class(year_ago_change))
    return {
        "company_majority": summarize(majority_predictions),
        "company_seasonal": summarize(seasonal_predictions),
        "previous_quarter_class": summarize(previous_predictions),
        "same_quarter_last_year_class": summarize(year_ago_predictions),
    }


def main():
    args = parse_arguments()
    company_names = [value.strip() for value in args.companies.split(",") if value.strip()]
    architectures = [value.strip() for value in args.architectures.split(",") if value.strip()]
    for company in company_names:
        if company not in COMPANIES:
            raise ValueError("Unknown company %s" % company)
    for architecture in architectures:
        if architecture not in EXPERIMENT_ARCHITECTURES:
            raise ValueError("Unknown architecture %s" % architecture)
    if args.validation_year >= args.test_year:
        raise ValueError("validation-year must precede test-year")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)
    records = build_records(company_names, args, device)
    train_records, validation_records, test_records = split_records(
        records, args.validation_year, args.test_year)
    print("Independent quarters: train %d, validation %d, test %d"
          % (len(train_records), len(validation_records), len(test_records)))
    print("Target: current-quarter four-class percent_change; financial input: t-1..t-%d"
          % args.lookback)

    baselines = baseline_metrics(train_records + validation_records, test_records)
    print("Baselines:", json.dumps(baselines, indent=2))
    results = {
        "target": "current-quarter percent_change class (0 decrease, 1 0-15%, 2 15-30%, 3 >30%)",
        "data_scope": "local tweet CSVs and local quarterly financial CSVs only",
        "financial_features": list(FINANCIAL_FEATURE_NAMES),
        "companies": company_names,
        "split": {
            "train": sorted(set(record.quarter for record in train_records)),
            "validation": sorted(set(record.quarter for record in validation_records)),
            "test": sorted(set(record.quarter for record in test_records)),
            "independent_company_quarters": {
                "train": len(train_records),
                "validation": len(validation_records),
                "test": len(test_records),
            },
        },
        "text_sampling": {
            "bins": args.bins,
            "tweets_per_bin": args.tweets_per_bin,
            "variants": args.variants,
        },
        "baselines": baselines,
        "models": {},
    }

    for architecture in architectures:
        print("\n===", architecture, "===")
        run_results, run_probabilities = [], []
        for run in range(args.runs):
            run_seed = args.seed + run * 100
            print("Run %d/%d, seed %d" % (run + 1, args.runs, run_seed))
            run_train, run_validation, run_test = train_records, validation_records, test_records
            if architecture == "fusion-shuffled-text":
                run_train = shuffle_text_within_company(train_records, run_seed + 1)
                run_validation = shuffle_text_within_company(validation_records, run_seed + 2)
                run_test = shuffle_text_within_company(test_records, run_seed + 3)
            best_epoch, validation_loss = fit_with_validation(
                architecture, run_train, run_validation, args, device, run_seed)
            print("Selected epoch %d; refitting on train + validation" % best_epoch)
            metrics, probabilities = refit_and_test(
                architecture, run_train, run_validation, run_test,
                best_epoch, args, device, run_seed)
            metrics["seed"] = run_seed
            metrics["selected_epoch"] = best_epoch
            metrics["validation_loss"] = validation_loss
            run_results.append(metrics)
            run_probabilities.append(probabilities)
            print("Test accuracy %.4f, MCC %.4f" % (metrics["accuracy"], metrics["mcc"]))

        ensemble_probabilities = np.mean(run_probabilities, axis=0)
        ensemble = metric_summary(test_records, ensemble_probabilities)
        accuracies = [result["accuracy"] for result in run_results]
        mccs = [result["mcc"] for result in run_results]
        results["models"][architecture] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "mcc_mean": float(np.mean(mccs)),
            "mcc_std": float(np.std(mccs)),
            "ensemble": ensemble,
            "runs": run_results,
        }
        print("%s mean accuracy %.4f (+/- %.4f), mean MCC %.4f (+/- %.4f)"
              % (architecture, np.mean(accuracies), np.std(accuracies),
                 np.mean(mccs), np.std(mccs)))
        print("%s ensemble accuracy %.4f, MCC %.4f"
              % (architecture, ensemble["accuracy"], ensemble["mcc"]))

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
