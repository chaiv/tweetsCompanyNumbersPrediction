"""Ablate hierarchical Top2Vec, MiniLM semantics, safe metadata and seasonal fusion."""

import argparse
import json
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader

from classifier.MultiViewQuarterDataset import MultiViewQuarterGroupDataset, SAFE_METADATA_FEATURE_NAMES
from classifier.MultiViewQuarterModel import MultiViewQuarterClassifier
from classifier.QuarterAlignedDataset import build_quarter_groups, select_balanced_quarter_groups
from nlpvectors.TweetTokenizer import TweetTokenizer
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from trainQuarterAlignedEmbeddingModel import (
    EXPERIMENTS,
    baseline_results,
    calculate_seasonal_log_prior,
    class_weights,
    seed_everything,
    summarize_predictions,
)
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter


ARCHITECTURES = {
    "metadata": dict(use_top2vec=False, use_sentence=False, use_metadata=True, seasonal=False),
    "top2vec": dict(use_top2vec=True, use_sentence=False, use_metadata=False, seasonal=False),
    "minilm": dict(use_top2vec=False, use_sentence=True, use_metadata=False, seasonal=False),
    "fusion": dict(use_top2vec=True, use_sentence=True, use_metadata=True, seasonal=False),
    "fusion-seasonal": dict(use_top2vec=True, use_sentence=True, use_metadata=True, seasonal=True),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="apple-eps")
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=256)
    parser.add_argument("--test-groups-per-quarter", type=int, default=512)
    parser.add_argument("--max-words-per-tweet", type=int, default=48)
    parser.add_argument("--sentence-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", default="../../output/multiview_quarter_results.json")
    return parser.parse_args()


def collate_batch(batch, pad_token_idx):
    word_groups, sentences, metadata, labels, quarters = zip(*batch)
    tweets_per_group = len(word_groups[0])
    flattened_words = [tweet for group in word_groups for tweet in group]
    padded_words = pad_sequence(
        flattened_words, batch_first=True, padding_value=pad_token_idx)
    padded_words = padded_words.reshape(len(batch), tweets_per_group, -1)
    return (
        padded_words,
        torch.stack(sentences),
        torch.stack(metadata),
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


def metadata_statistics(datasets):
    metadata = torch.cat([dataset.metadata for dataset in datasets], dim=0)
    return metadata.mean(dim=0).numpy(), metadata.std(dim=0).clamp(min=1e-6).numpy()


def create_model(architecture, vectors, pad_token_idx, num_classes, metadata_size,
                 metadata_mean, metadata_std, sentence_size, labels, quarters):
    config = ARCHITECTURES[architecture]
    seasonal_prior = None
    if config["seasonal"]:
        seasonal_prior = calculate_seasonal_log_prior(labels, quarters, num_classes)
    return MultiViewQuarterClassifier(
        num_classes=num_classes,
        metadata_size=metadata_size,
        metadata_mean=metadata_mean,
        metadata_std=metadata_std,
        use_top2vec=config["use_top2vec"],
        use_sentence=config["use_sentence"],
        use_metadata=config["use_metadata"],
        word_vectors=vectors,
        pad_token_idx=pad_token_idx,
        sentence_embedding_size=sentence_size,
        seasonal_log_prior=seasonal_prior,
    )


def train_epoch(model, loader, optimizer, loss_function, device, scaler):
    model.train()
    losses = []
    for words, sentences, metadata, labels, quarters in loader:
        words = words.to(device, non_blocking=True)
        sentences = sentences.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        calendar_quarters = torch.tensor(
            [int(quarter[-1]) - 1 for quarter in quarters], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(words, sentences, metadata, calendar_quarters)
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
    for words, sentences, metadata, batch_labels, batch_quarters in loader:
        words = words.to(device, non_blocking=True)
        sentences = sentences.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        calendar_quarters = torch.tensor(
            [int(quarter[-1]) - 1 for quarter in batch_quarters],
            dtype=torch.long, device=device)
        logits = model(words, sentences, metadata, calendar_quarters)
        losses.append(float(loss_function(logits, batch_labels).cpu()))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.extend(batch_labels.cpu().numpy().tolist())
        quarters.extend(batch_quarters)
    return float(np.mean(losses)), np.concatenate(probabilities), np.asarray(labels), quarters


def fit_with_validation(architecture, vectors, pad_token_idx, num_classes, train_dataset,
                        validation_dataset, args, device):
    metadata_mean, metadata_std = metadata_statistics([train_dataset])
    model = create_model(
        architecture, vectors, pad_token_idx, num_classes, train_dataset.metadata.shape[1],
        metadata_mean, metadata_std, train_dataset.sentence_embeddings.shape[-1],
        train_dataset.labels, train_dataset.quarters).to(device)
    train_loader = make_loader(train_dataset, args.batch_size, pad_token_idx, shuffle=True)
    validation_loader = make_loader(
        validation_dataset, args.batch_size, pad_token_idx, shuffle=False)
    weights = class_weights(train_dataset.labels, num_classes, device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=4e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_loss, best_epoch, stale_epochs = float("inf"), 1, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        validation_loss, probabilities, labels, quarters = predict(
            model, validation_loader, loss_function, device)
        summary = summarize_predictions(probabilities, labels, quarters)
        print("  epoch %02d train_loss %.4f val_loss %.4f val_quarter_acc %.3f"
              % (epoch, train_loss, validation_loss, summary["quarter"]["accuracy"]))
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    del model
    torch.cuda.empty_cache()
    return best_epoch, best_loss


def refit_and_test(architecture, vectors, pad_token_idx, num_classes, train_dataset,
                   validation_dataset, test_dataset, epochs, args, device):
    seed_everything(args.seed)
    combined_labels = train_dataset.labels + validation_dataset.labels
    combined_quarters = train_dataset.quarters + validation_dataset.quarters
    metadata_mean, metadata_std = metadata_statistics([train_dataset, validation_dataset])
    model = create_model(
        architecture, vectors, pad_token_idx, num_classes, train_dataset.metadata.shape[1],
        metadata_mean, metadata_std, train_dataset.sentence_embeddings.shape[-1],
        combined_labels, combined_quarters).to(device)
    combined = ConcatDataset([train_dataset, validation_dataset])
    combined.labels = combined_labels
    loader = make_loader(combined, args.batch_size, pad_token_idx, shuffle=True)
    test_loader = make_loader(test_dataset, args.batch_size, pad_token_idx, shuffle=False)
    weights = class_weights(combined_labels, num_classes, device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=4e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, loss_function, device, scaler)
        print("  refit epoch %02d/%02d loss %.4f" % (epoch, epochs, loss))
    test_loss, probabilities, labels, quarters = predict(
        model, test_loader, loss_function, device)
    result = summarize_predictions(probabilities, labels, quarters)
    result["test_loss"] = test_loss
    result["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    prediction_path = EXPERIMENTS[args.experiment]
    validation_year = args.test_year - 1
    print("Device:", device)
    frame = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["tweet_id", "writer", "post_date", "body", "class"],
    )
    frame["body"] = frame["body"].fillna("")
    frame, groups = build_quarter_groups(frame, prediction_path.getTweetGroupSize())
    all_quarters = sorted({group.quarter for group in groups})
    train_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) < validation_year]
    validation_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) == validation_year]
    test_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) == args.test_year]
    print("Train:", train_quarters)
    print("Validation:", validation_quarters)
    print("Test:", test_quarters)

    train_groups = select_balanced_quarter_groups(
        groups, train_quarters, args.groups_per_quarter, seed=args.seed)
    validation_groups = select_balanced_quarter_groups(
        groups, validation_quarters, args.groups_per_quarter, seed=args.seed + 1)
    test_groups = select_balanced_quarter_groups(
        groups, test_quarters, args.test_groups_per_quarter, seed=args.seed + 2)
    print("Selected groups: train %d, validation %d, test %d"
          % (len(train_groups), len(validation_groups), len(test_groups)))

    print("Loading Top2Vec word vectors")
    word_vectors = KeyedVectors.load_word2vec_format(
        prediction_path.getWordVectorsPath(), binary=False)
    encoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())
    print("Loading frozen sentence encoder", args.sentence_model)
    sentence_model = SentenceTransformer(args.sentence_model, device=str(device))
    datasets = []
    for name, selected_groups in (
        ("train", train_groups),
        ("validation", validation_groups),
        ("test", test_groups),
    ):
        print("Encoding", name)
        datasets.append(MultiViewQuarterGroupDataset(
            frame,
            selected_groups,
            tokenizer,
            encoder,
            sentence_model,
            max_words_per_tweet=args.max_words_per_tweet,
            sentence_batch_size=args.sentence_batch_size,
        ))
    train_dataset, validation_dataset, test_dataset = datasets
    del sentence_model
    torch.cuda.empty_cache()

    num_classes = prediction_path.getPredictionClassMapper().get_number_of_classes()
    baselines = baseline_results(
        train_groups + validation_groups, test_dataset, num_classes)
    results = {
        "experiment": args.experiment,
        "metadata_features": list(SAFE_METADATA_FEATURE_NAMES),
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
        "baselines": baselines,
        "models": {},
    }
    print("Baselines:", json.dumps(baselines, indent=2))

    architectures = [value.strip() for value in args.architectures.split(",") if value.strip()]
    for architecture in architectures:
        if architecture not in ARCHITECTURES:
            raise ValueError("Unknown architecture %s" % architecture)
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
