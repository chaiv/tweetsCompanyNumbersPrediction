"""Retrain the dissertation-era Amazon@10 binary LSTM protocol.

The protocol is preserved for reproducibility only.  Its shuffled, label-aware
tweet grouping must not be reported as a future-quarter forecasting result.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Subset
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split

from PredictionModelPath import AMAZON_REVENUE_10_LSTM_BINARY_CLASS
from classifier.CreateClassifierModel import CreateClassifierModel
from classifier.ModelEvaluationHelper import loadModel
from classifier.Trainer import Trainer
from classifier.TweetGroupDataset import TweetGroupDataset
from classifier.transformer.DatasetUtils import createDataloader
from nlpvectors.DataframeSplitter import DataframeSplitter
from nlpvectors.TweetTokenizer import TweetTokenizer
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from tweetpreprocess.EqualClassSampler import EqualClassSampler
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--lstm-dropout",
        type=float,
        default=0.2,
        help="Table 12 specifies 0.2; pass 0.0 to reproduce the old constructor literally.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/old-model-reproduction/retrained-10fold-dropout-0.2-seed-1337"),
    )
    parser.add_argument(
        "--reuse-model-across-folds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse one model across folds. Disabled for independent cross-validation folds.",
    )
    return parser.parse_args()


def evaluate_checkpoint(path, word_vectors, loader, device, lstm_dropout):
    model = loadModel(
        path,
        word_vectors,
        num_classes=2,
        device=device,
        lstmDropout=lstm_dropout,
    )
    true_labels = []
    predictions = []
    with torch.inference_mode():
        for inputs, labels in loader:
            outputs = model(inputs.to(device, non_blocking=True))
            predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            true_labels.extend(labels.tolist())
    return {
        "samples": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "mcc": float(matthews_corrcoef(true_labels, predictions)),
    }


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = AMAZON_REVENUE_10_LSTM_BINARY_CLASS
    dataframe = pd.read_csv(config.getDataframePath()).fillna("")
    dataframe = EqualClassSampler().getDfWithEqualNumberOfClassSamples(dataframe)
    splitter = DataframeSplitter()
    groups = splitter.getSplitIds(dataframe, config.getTweetGroupSize())
    folds = list(KFold(n_splits=args.folds, shuffle=True, random_state=1337).split(groups))
    word_vectors = KeyedVectors.load_word2vec_format(config.getWordVectorsPath(), binary=False)
    encoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())

    result = {
        "protocol": "historical_mixed_group_cv_retraining",
        "warning": "This label-aware shuffled evaluation is not a future-quarter forecast.",
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "epochs": args.epochs,
        "folds": args.folds,
        "tweet_group_size": config.getTweetGroupSize(),
        "balanced_tweets": len(dataframe),
        "tweet_groups": len(groups),
        "reuse_model_across_folds": args.reuse_model_across_folds,
        "lstm_dropout": args.lstm_dropout,
        "fold_results": [],
    }
    result_path = args.output_dir / "metrics.json"
    if args.start_fold > 0 and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        result["fold_results"] = [
            item for item in previous.get("fold_results", []) if item["fold"] < args.start_fold
        ]
    full_dataset = None
    if args.start_fold < args.folds:
        full_dataset = TweetGroupDataset(
            dataframe,
            groups,
            np.arange(len(groups)),
            tokenizer,
            encoder,
        )
    model = None

    for fold, (train_indexes, test_indexes) in enumerate(folds):
        if fold < args.start_fold:
            continue
        # Give every independent fold the same initialization and shuffled-loader seed.
        # Otherwise a failed fold can be caused solely by the advanced global RNG state.
        if not args.reuse_model_across_folds:
            pl.seed_everything(args.seed, workers=True)
        if model is None or not args.reuse_model_across_folds:
            model = CreateClassifierModel(
                word_vectors=word_vectors,
                num_classes=2,
                lstmDropout=args.lstm_dropout,
            ).createModel()
        train_indexes, validation_indexes = train_test_split(
            train_indexes, random_state=1337, test_size=0.3
        )
        np.save(args.output_dir / f"test_idx_fold{fold}.npy", test_indexes)
        print(
            f"fold={fold} train={len(train_indexes)} validation={len(validation_indexes)} "
            f"test={len(test_indexes)} device={device}",
            flush=True,
        )

        train_data = Subset(full_dataset, train_indexes.tolist())
        validation_data = Subset(full_dataset, validation_indexes.tolist())
        test_data = Subset(full_dataset, test_indexes.tolist())
        best_path = Trainer().train(
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            pad_token_idx=encoder.getPADTokenID(),
            model=model,
            train_data=train_data,
            val_data=validation_data,
            test_data=test_data,
            loggerPath=str(args.output_dir / "logs"),
            loggerName="amazon-lstm-n10-binary",
            checkpointPath=str(args.output_dir),
            checkpointName=f"tweetpredict_fold{fold}",
        )
        test_loader = createDataloader(
            test_data, args.batch_size, args.num_workers, encoder.getPADTokenID(), shuffle=False
        )
        metrics = evaluate_checkpoint(
            best_path, word_vectors, test_loader, device, args.lstm_dropout
        )
        fold_result = {"fold": fold, "checkpoint": best_path, **metrics}
        result["fold_results"].append(fold_result)
        write_result(result_path, result)
        print(
            f"fold={fold} accuracy={metrics['accuracy']:.6f} mcc={metrics['mcc']:.6f}",
            flush=True,
        )

        del train_data, validation_data, test_data, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["mean_accuracy"] = float(
        np.mean([item["accuracy"] for item in result["fold_results"]])
    )
    result["mean_mcc"] = float(np.mean([item["mcc"] for item in result["fold_results"]]))
    result["std_accuracy"] = float(
        np.std([item["accuracy"] for item in result["fold_results"]])
    )
    result["std_mcc"] = float(np.std([item["mcc"] for item in result["fold_results"]]))
    result["collapsed_folds"] = [
        item["fold"]
        for item in result["fold_results"]
        if item["accuracy"] < 0.55 or item["mcc"] < 0.05
    ]
    result["closest_fold_to_published_target"] = min(
        result["fold_results"],
        key=lambda item: (item["accuracy"] - 0.87) ** 2 + (item["mcc"] - 0.77) ** 2,
    )["fold"]
    write_result(result_path, result)
    print(
        f"mean accuracy={result['mean_accuracy']:.6f} mcc={result['mean_mcc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
