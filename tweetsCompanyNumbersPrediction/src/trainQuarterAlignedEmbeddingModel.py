"""Train honest quarter-aligned Top2Vec text baselines and a compact attention BiLSTM.

The experiment uses 2015-2017 for fitting, 2018 for selecting the epoch count, refits on
2015-2018, and evaluates once on 2019 by default.  Groups are built within reporting quarters and
sampled equally per quarter.  Results are reported both per tweet group and per independent quarter,
alongside no-text majority and seasonal baselines.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader

from PredictionModelPath import (
    AMAZON_REVENUE_10_LSTM_BINARY_CLASS,
    AMAZON_REVENUE_10_LSTM_MULTI_CLASS,
    APPLE__EPS_10_LSTM_MULTI_CLASS,
    TESLA_CAR_SALES_10_LSTM_MULTI_CLASS,
)
from classifier.QuarterAlignedDataset import (
    EncodedQuarterGroupDataset,
    build_quarter_groups,
    select_balanced_quarter_groups,
)
from classifier.QuarterTextModels import (
    MeanEmbeddingClassifier,
    PackedAttentionLSTMClassifier,
    SeasonalResidualEmbeddingClassifier,
    count_trainable_parameters,
)
from nlpvectors.TweetTokenizer import TweetTokenizer
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter


EXPERIMENTS = {
    "apple-eps": APPLE__EPS_10_LSTM_MULTI_CLASS,
    "amazon-revenue-binary": AMAZON_REVENUE_10_LSTM_BINARY_CLASS,
    "amazon-revenue-4class": AMAZON_REVENUE_10_LSTM_MULTI_CLASS,
    "tesla-sales": TESLA_CAR_SALES_10_LSTM_MULTI_CLASS,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="apple-eps")
    parser.add_argument("--architectures", default="mean,bilstm,hybrid",
                        help="Comma-separated subset of: mean,bilstm,hybrid")
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=512)
    parser.add_argument("--test-groups-per-quarter", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="../../output/quarter_aligned_embedding_results.json")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch, pad_token_idx):
    sequences, labels, quarters = zip(*batch)
    return (
        pad_sequence(sequences, batch_first=True, padding_value=pad_token_idx),
        torch.tensor(labels, dtype=torch.long),
        list(quarters),
    )


def make_loader(dataset, batch_size, pad_token_idx, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_batch, pad_token_idx=pad_token_idx),
    )


def class_weights(labels, num_classes, device):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, device=device)


def train_epoch(model, loader, optimizer, loss_function, device, scaler):
    model.train()
    losses = []
    for tokens, labels, quarters in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        calendar_quarters = torch.tensor(
            [int(quarter[-1]) - 1 for quarter in quarters], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(tokens, calendar_quarters)
            loss = loss_function(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, loss_function, device):
    model.eval()
    losses, probabilities, labels, quarters = [], [], [], []
    for tokens, batch_labels, batch_quarters in loader:
        tokens = tokens.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        calendar_quarters = torch.tensor(
            [int(quarter[-1]) - 1 for quarter in batch_quarters],
            dtype=torch.long, device=device)
        logits = model(tokens, calendar_quarters)
        losses.append(float(loss_function(logits, batch_labels).cpu()))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.extend(batch_labels.cpu().numpy().tolist())
        quarters.extend(batch_quarters)
    return float(np.mean(losses)), np.concatenate(probabilities), np.asarray(labels), quarters


def calculate_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "n": int(len(labels)),
    }


def summarize_predictions(probabilities, labels, quarters):
    group_predictions = probabilities.argmax(axis=1)
    group_metrics = calculate_metrics(labels, group_predictions)
    by_quarter = defaultdict(list)
    true_by_quarter = {}
    for probability, label, quarter in zip(probabilities, labels, quarters):
        by_quarter[quarter].append(probability)
        true_by_quarter.setdefault(quarter, int(label))
        if true_by_quarter[quarter] != int(label):
            raise ValueError("Quarter %s contains multiple target labels" % quarter)
    ordered_quarters = sorted(by_quarter)
    quarter_predictions = np.asarray([
        np.mean(by_quarter[quarter], axis=0).argmax() for quarter in ordered_quarters])
    quarter_labels = np.asarray([true_by_quarter[quarter] for quarter in ordered_quarters])
    return {
        "group": group_metrics,
        "quarter": calculate_metrics(quarter_labels, quarter_predictions),
        "per_quarter": {
            quarter: {"true": int(label), "predicted": int(prediction)}
            for quarter, label, prediction in zip(ordered_quarters, quarter_labels, quarter_predictions)
        },
    }


def baseline_results(training_groups, test_dataset, num_classes):
    quarter_labels = {}
    for group in training_groups:
        quarter_labels[group.quarter] = group.label
    majority = Counter(quarter_labels.values()).most_common(1)[0][0]
    seasonal_map = {}
    for calendar_quarter in range(1, 5):
        values = [label for quarter, label in quarter_labels.items()
                  if int(quarter[-1]) == calendar_quarter]
        seasonal_map[calendar_quarter] = Counter(values).most_common(1)[0][0]

    labels = np.asarray(test_dataset.labels)
    quarters = test_dataset.quarters
    baseline_predictions = {
        "majority": np.full(len(labels), majority),
        "seasonal": np.asarray([seasonal_map[int(quarter[-1])] for quarter in quarters]),
    }
    results = {}
    for name, predictions in baseline_predictions.items():
        probabilities = np.eye(num_classes, dtype=np.float32)[predictions.astype(int)]
        results[name] = summarize_predictions(probabilities, labels, quarters)
    return results


def calculate_seasonal_log_prior(labels, quarters, num_classes, smoothing=0.5):
    """Smoothed class probabilities per calendar quarter, counting reporting quarters once."""
    label_by_quarter = {}
    for label, quarter in zip(labels, quarters):
        label_by_quarter.setdefault(quarter, int(label))
        if label_by_quarter[quarter] != int(label):
            raise ValueError("Quarter %s contains multiple target labels" % quarter)
    counts = np.full((4, num_classes), smoothing, dtype=np.float32)
    for quarter, label in label_by_quarter.items():
        counts[int(quarter[-1]) - 1, label] += 1.0
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probabilities)


def create_model(architecture, vectors, pad_token_idx, num_classes, seasonal_log_prior=None):
    if architecture == "mean":
        return MeanEmbeddingClassifier(vectors, pad_token_idx, num_classes)
    if architecture == "bilstm":
        return PackedAttentionLSTMClassifier(vectors, pad_token_idx, num_classes)
    if architecture == "hybrid":
        if seasonal_log_prior is None:
            raise ValueError("hybrid architecture requires a seasonal_log_prior")
        return SeasonalResidualEmbeddingClassifier(
            vectors, pad_token_idx, num_classes, seasonal_log_prior)
    raise ValueError("Unknown architecture %s" % architecture)


def fit_with_validation(architecture, vectors, pad_token_idx, num_classes, train_dataset,
                        validation_dataset, args, device):
    prior = calculate_seasonal_log_prior(
        train_dataset.labels, train_dataset.quarters, num_classes)
    model = create_model(
        architecture, vectors, pad_token_idx, num_classes, seasonal_log_prior=prior).to(device)
    train_loader = make_loader(train_dataset, args.batch_size, pad_token_idx, shuffle=True)
    validation_loader = make_loader(validation_dataset, args.batch_size, pad_token_idx, shuffle=False)
    weights = class_weights(train_dataset.labels, num_classes, device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    learning_rate = 3e-4 if architecture == "bilstm" else 1e-3
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_loss, best_epoch, stale_epochs = float("inf"), 1, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        validation_loss, probabilities, labels, quarters = predict(
            model, validation_loader, loss_function, device)
        validation_summary = summarize_predictions(probabilities, labels, quarters)
        print("  epoch %02d train_loss %.4f val_loss %.4f val_quarter_acc %.3f"
              % (epoch, train_loss, validation_loss, validation_summary["quarter"]["accuracy"]))
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    return best_epoch, best_loss


def refit_and_test(architecture, vectors, pad_token_idx, num_classes, train_dataset,
                   validation_dataset, test_dataset, epochs, args, device):
    seed_everything(args.seed)
    combined_labels = train_dataset.labels + validation_dataset.labels
    combined_quarters = train_dataset.quarters + validation_dataset.quarters
    prior = calculate_seasonal_log_prior(combined_labels, combined_quarters, num_classes)
    model = create_model(
        architecture, vectors, pad_token_idx, num_classes, seasonal_log_prior=prior).to(device)
    combined_dataset = ConcatDataset([train_dataset, validation_dataset])
    combined_dataset.labels = combined_labels
    loader = make_loader(combined_dataset, args.batch_size, pad_token_idx, shuffle=True)
    test_loader = make_loader(test_dataset, args.batch_size, pad_token_idx, shuffle=False)
    weights = class_weights(combined_dataset.labels, num_classes, device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    learning_rate = 3e-4 if architecture == "bilstm" else 1e-3
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, loss_function, device, scaler)
        print("  refit epoch %02d/%02d loss %.4f" % (epoch, epochs, loss))
    test_loss, probabilities, labels, quarters = predict(model, test_loader, loss_function, device)
    result = summarize_predictions(probabilities, labels, quarters)
    result["test_loss"] = test_loss
    result["trainable_parameters"] = count_trainable_parameters(model)
    return result


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    prediction_path = EXPERIMENTS[args.experiment]
    validation_year = args.test_year - 1
    print("Device:", device)
    print("Reading", prediction_path.getDataframePath())
    frame = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["tweet_id", "post_date", "body", "class"],
    )
    frame["body"] = frame["body"].fillna("")
    frame, groups = build_quarter_groups(frame, prediction_path.getTweetGroupSize())
    all_quarters = sorted({group.quarter for group in groups})
    train_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) < validation_year]
    validation_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) == validation_year]
    test_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) == args.test_year]
    if not train_quarters or not validation_quarters or not test_quarters:
        raise ValueError("Need non-empty train, validation and test years")
    print("Train quarters:", train_quarters)
    print("Validation quarters:", validation_quarters)
    print("Test quarters:", test_quarters)

    train_groups = select_balanced_quarter_groups(
        groups, train_quarters, args.groups_per_quarter, seed=args.seed)
    validation_groups = select_balanced_quarter_groups(
        groups, validation_quarters, args.groups_per_quarter, seed=args.seed + 1)
    test_groups = select_balanced_quarter_groups(
        groups, test_quarters, args.test_groups_per_quarter, seed=args.seed + 2)
    print("Selected groups: train %d, validation %d, test %d"
          % (len(train_groups), len(validation_groups), len(test_groups)))

    print("Loading pretrained Top2Vec word vectors")
    word_vectors = KeyedVectors.load_word2vec_format(
        prediction_path.getWordVectorsPath(), binary=False)
    encoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())
    print("Encoding selected groups")
    train_dataset = EncodedQuarterGroupDataset(
        frame, train_groups, tokenizer, encoder, max_tokens=args.max_tokens)
    validation_dataset = EncodedQuarterGroupDataset(
        frame, validation_groups, tokenizer, encoder, max_tokens=args.max_tokens)
    test_dataset = EncodedQuarterGroupDataset(
        frame, test_groups, tokenizer, encoder, max_tokens=args.max_tokens)

    num_classes = prediction_path.getPredictionClassMapper().get_number_of_classes()
    baseline = baseline_results(train_groups + validation_groups, test_dataset, num_classes)
    print("Baselines:", json.dumps(baseline, indent=2))
    results = {
        "experiment": args.experiment,
        "split": {
            "train_quarters": train_quarters,
            "validation_quarters": validation_quarters,
            "test_quarters": test_quarters,
            "groups": {
                "train": len(train_groups),
                "validation": len(validation_groups),
                "test": len(test_groups),
            },
        },
        "baselines": baseline,
        "models": {},
    }

    architectures = [value.strip() for value in args.architectures.split(",") if value.strip()]
    for architecture in architectures:
        if architecture not in {"mean", "bilstm", "hybrid"}:
            raise ValueError("Architectures must be a subset of mean,bilstm,hybrid")
        print("\n===", architecture, "===")
        seed_everything(args.seed)
        best_epoch, validation_loss = fit_with_validation(
            architecture,
            word_vectors.vectors,
            encoder.getPADTokenID(),
            num_classes,
            train_dataset,
            validation_dataset,
            args,
            device,
        )
        print("Best epoch %d; refitting on train + validation" % best_epoch)
        result = refit_and_test(
            architecture,
            word_vectors.vectors,
            encoder.getPADTokenID(),
            num_classes,
            train_dataset,
            validation_dataset,
            test_dataset,
            best_epoch,
            args,
            device,
        )
        result["selected_epoch"] = best_epoch
        result["validation_loss"] = validation_loss
        results["models"][architecture] = result
        print(json.dumps(result, indent=2))

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
