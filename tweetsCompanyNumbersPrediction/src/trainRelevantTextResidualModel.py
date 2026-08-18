"""Future-only evaluation of relevant Top2Vec text as a financial residual."""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import ConcatDataset, DataLoader

from classifier.ExplainableResidualQuarterModel import ExplainableResidualQuarterModel
from classifier.QuarterAlignedDataset import reporting_quarters
from classifier.RelevantQuarterTextDataset import (
    RELEVANCE_ANCHORS,
    RelevantQuarterDataset,
    build_relevant_quarter_records,
    select_relevant_tweet_pools,
    shuffle_record_text,
)
from nlpvectors.TweetTokenizer import TweetTokenizer
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from trainQuarterAlignedEmbeddingModel import EXPERIMENTS
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(RELEVANCE_ANCHORS), default="apple-eps")
    parser.add_argument("--test-years", default="2017,2018,2019")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--pool-per-bin", type=int, default=48)
    parser.add_argument("--max-candidates-per-quarter", type=int, default=3072)
    parser.add_argument("--variants", type=int, default=12)
    parser.add_argument("--tweets-per-bin", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=40)
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sentence-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modality-dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="../../output/relevant_text_residual_results.json")
    parser.add_argument("--checkpoint-dir", default="../../output/relevant_text_checkpoints")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_records(records, validation_year, test_year):
    return (
        [record for record in records if int(record.quarter[:4]) < validation_year],
        [record for record in records if int(record.quarter[:4]) == validation_year],
        [record for record in records if int(record.quarter[:4]) == test_year],
    )


def make_loader(dataset, args, shuffle):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def create_model(vectors, pad_token_idx, records, args, device):
    return ExplainableResidualQuarterModel(
        vectors,
        pad_token_idx,
        financial_feature_size=records[0].financial_sequence.shape[-1],
        hidden_size=args.hidden_size,
        modality_dropout=args.modality_dropout,
    ).to(device)


def class_weights(records, device):
    counts = np.bincount([record.label for record in records], minlength=4).astype(np.float32)
    weights = len(records) / np.maximum(counts, 1.0)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    losses = []
    for words, financial, _, labels, _ in loader:
        words, financial, labels = words.to(device), financial.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.all_logits(words, financial, apply_modality_dropout=True)
        loss = (loss_function(logits["fusion"], labels)
                + 0.35 * loss_function(logits["text"], labels)
                + 0.50 * loss_function(logits["finance"], labels))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, dataset, args, loss_function, device):
    model.eval()
    loader = make_loader(dataset, args, shuffle=False)
    probabilities = {name: [[] for _ in dataset.records]
                     for name in ("finance", "text", "fusion")}
    losses = []
    for words, financial, _, labels, record_indexes in loader:
        words, financial, labels = words.to(device), financial.to(device), labels.to(device)
        logits = model.all_logits(words, financial, apply_modality_dropout=False)
        losses.append(float(loss_function(logits["fusion"], labels).cpu()))
        for name in probabilities:
            batch_probabilities = torch.softmax(logits[name], dim=1).cpu().numpy()
            for values, record_index in zip(batch_probabilities, record_indexes.tolist()):
                probabilities[name][record_index].append(values)
    return float(np.mean(losses)), {
        name: np.asarray([np.mean(values, axis=0) for values in per_record])
        for name, per_record in probabilities.items()
    }


def metric_summary(records, probabilities):
    labels = np.asarray([record.label for record in records])
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "true": labels.tolist(),
        "predicted": predictions.tolist(),
        "quarters": [record.quarter for record in records],
    }


def fit_with_validation(vectors, pad_token_idx, train_records, validation_records,
                        args, device, seed):
    seed_everything(seed)
    model = create_model(vectors, pad_token_idx, train_records + validation_records, args, device)
    train_dataset = RelevantQuarterDataset(train_records)
    validation_dataset = RelevantQuarterDataset(validation_records)
    loader = make_loader(train_dataset, args, shuffle=True)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights(train_records, device), label_smoothing=0.03)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-4, weight_decay=2e-3)
    best_loss, best_epoch, stale = float("inf"), 1, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, loss_function, device)
        validation_loss, probabilities = predict(
            model, validation_dataset, args, loss_function, device)
        metrics = metric_summary(validation_records, probabilities["fusion"])
        print("  epoch %02d loss %.4f val_loss %.4f val_acc %.3f val_mcc %.3f"
              % (epoch, train_loss, validation_loss, metrics["accuracy"], metrics["mcc"]))
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    del model
    torch.cuda.empty_cache()
    return best_epoch, best_loss


def refit_and_predict(vectors, pad_token_idx, train_records, validation_records, test_records,
                      epochs, args, device, seed, checkpoint_path=None):
    seed_everything(seed)
    combined_records = train_records + validation_records
    model = create_model(vectors, pad_token_idx, combined_records, args, device)
    combined_dataset = RelevantQuarterDataset(combined_records)
    test_dataset = RelevantQuarterDataset(test_records)
    shuffled_dataset = RelevantQuarterDataset(shuffle_record_text(test_records, seed + 17))
    loader = make_loader(combined_dataset, args, shuffle=True)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights(combined_records, device), label_smoothing=0.03)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-4, weight_decay=2e-3)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, loss_function, device)
        print("  refit %02d/%02d loss %.4f" % (epoch, epochs, loss))
    _, probabilities = predict(model, test_dataset, args, loss_function, device)
    _, shuffled_probabilities = predict(model, shuffled_dataset, args, loss_function, device)
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "hidden_size": args.hidden_size,
            "modality_dropout": args.modality_dropout,
            "pad_token_idx": pad_token_idx,
            "financial_feature_size": combined_records[0].financial_sequence.shape[-1],
        }, checkpoint_path)
    scales = {
        "finance": float(torch.nn.functional.softplus(model.finance_residual_scale).detach().cpu()),
        "text": float(torch.nn.functional.softplus(model.text_residual_scale).detach().cpu()),
    }
    del model
    torch.cuda.empty_cache()
    return probabilities, shuffled_probabilities["fusion"], scales


def main():
    args = parse_arguments()
    test_years = [int(value.strip()) for value in args.test_years.split(",") if value.strip()]
    if min(test_years) < 2017:
        raise ValueError("Rolling neural evaluation needs train and validation years before test")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    prediction_path = EXPERIMENTS[args.experiment]
    print("Device:", device)
    print("Experiment:", args.experiment)
    print("Protocol: strictly future test years", test_years)

    frame = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["tweet_id", "post_date", "body", "class"])
    frame["body"] = frame["body"].fillna("").astype(str)
    frame["reporting_quarter"] = reporting_quarters(frame["post_date"])
    frame.sort_values("post_date", kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    quarters = sorted({quarter for quarter in frame["reporting_quarter"].unique()
                       if 2015 <= int(quarter[:4]) <= max(test_years)})

    print("Loading Top2Vec vectors")
    word_vectors = KeyedVectors.load_word2vec_format(
        prediction_path.getWordVectorsPath(), binary=False)
    encoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())
    print("Loading frozen relevance encoder")
    sentence_model = SentenceTransformer(args.sentence_model, device=str(device))
    pools = select_relevant_tweet_pools(
        frame, quarters, sentence_model, args.experiment,
        bins=args.bins, pool_per_bin=args.pool_per_bin,
        max_candidates_per_quarter=args.max_candidates_per_quarter,
        sentence_batch_size=args.sentence_batch_size, seed=args.seed)
    del sentence_model
    torch.cuda.empty_cache()
    records = build_relevant_quarter_records(
        frame, pd.read_csv(prediction_path.getFinancialNumbersPath()), pools,
        tokenizer, encoder, variants=args.variants,
        tweets_per_bin=args.tweets_per_bin, max_words=args.max_words, seed=args.seed)
    print("Built %d independent quarter records, %d text bags each"
          % (len(records), args.variants))

    head_names = ("finance", "text", "fusion")
    probabilities = {name: [[] for _ in range(args.runs)] for name in head_names}
    shuffled_probabilities = [[] for _ in range(args.runs)]
    all_test_records, folds = [], []
    for test_year in test_years:
        validation_year = test_year - 1
        train_records, validation_records, test_records = split_records(
            records, validation_year, test_year)
        print("\n=== test %d: train < %d, validation %d ==="
              % (test_year, validation_year, validation_year))
        all_test_records.extend(test_records)
        fold = {"test_year": test_year, "runs": []}
        for run in range(args.runs):
            seed = args.seed + test_year + run * 100
            print("Run %d/%d seed %d" % (run + 1, args.runs, seed))
            best_epoch, validation_loss = fit_with_validation(
                word_vectors.vectors, encoder.getPADTokenID(), train_records,
                validation_records, args, device, seed)
            checkpoint = None
            if run == 0:
                checkpoint = os.path.abspath(os.path.join(
                    args.checkpoint_dir, "%s_%d.pt" % (args.experiment, test_year)))
            run_probabilities, shuffled, scales = refit_and_predict(
                word_vectors.vectors, encoder.getPADTokenID(), train_records,
                validation_records, test_records, best_epoch, args, device, seed, checkpoint)
            for name in head_names:
                probabilities[name][run].append(run_probabilities[name])
            shuffled_probabilities[run].append(shuffled)
            fold_metrics = {
                name: metric_summary(test_records, run_probabilities[name]) for name in head_names}
            fold_metrics["shuffled_text_fusion"] = metric_summary(test_records, shuffled)
            fold_metrics["selected_epoch"] = best_epoch
            fold_metrics["validation_loss"] = validation_loss
            fold_metrics["residual_scales"] = scales
            fold["runs"].append(fold_metrics)
            print("  test fusion acc %.3f MCC %.3f; text acc %.3f MCC %.3f"
                  % (fold_metrics["fusion"]["accuracy"], fold_metrics["fusion"]["mcc"],
                     fold_metrics["text"]["accuracy"], fold_metrics["text"]["mcc"]))
        folds.append(fold)

    models = {}
    for name in head_names:
        per_run = []
        combined = []
        for run in range(args.runs):
            values = np.concatenate(probabilities[name][run], axis=0)
            combined.append(values)
            per_run.append(metric_summary(all_test_records, values))
        ensemble = metric_summary(all_test_records, np.mean(combined, axis=0))
        models[name] = {
            "accuracy_mean": float(np.mean([value["accuracy"] for value in per_run])),
            "accuracy_std": float(np.std([value["accuracy"] for value in per_run])),
            "mcc_mean": float(np.mean([value["mcc"] for value in per_run])),
            "mcc_std": float(np.std([value["mcc"] for value in per_run])),
            "ensemble": ensemble,
            "runs": per_run,
        }
    shuffled_runs = []
    shuffled_combined = []
    for run in range(args.runs):
        values = np.concatenate(shuffled_probabilities[run], axis=0)
        shuffled_combined.append(values)
        shuffled_runs.append(metric_summary(all_test_records, values))
    models["shuffled_text_fusion"] = {
        "accuracy_mean": float(np.mean([value["accuracy"] for value in shuffled_runs])),
        "accuracy_std": float(np.std([value["accuracy"] for value in shuffled_runs])),
        "mcc_mean": float(np.mean([value["mcc"] for value in shuffled_runs])),
        "mcc_std": float(np.std([value["mcc"] for value in shuffled_runs])),
        "ensemble": metric_summary(all_test_records, np.mean(shuffled_combined, axis=0)),
        "runs": shuffled_runs,
    }

    for name, values in models.items():
        ensemble = values["ensemble"]
        print("%s: mean acc %.4f (+/- %.4f), MCC %.4f (+/- %.4f); "
              "ensemble acc %.4f MCC %.4f"
              % (name, values["accuracy_mean"], values["accuracy_std"],
                 values["mcc_mean"], values["mcc_std"],
                 ensemble["accuracy"], ensemble["mcc"]))

    results = {
        "protocol": "rolling-origin future-only",
        "experiment": args.experiment,
        "target": "current-quarter four-class percent_change",
        "text_horizon": "current-quarter tweets before financial result (nowcast)",
        "selection": {
            "label_independent": True,
            "anchors": list(RELEVANCE_ANCHORS[args.experiment]),
            "bins": args.bins,
            "pool_per_bin": args.pool_per_bin,
            "variants": args.variants,
            "tweets_per_bin": args.tweets_per_bin,
        },
        "independent_test_quarters": len(all_test_records),
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
