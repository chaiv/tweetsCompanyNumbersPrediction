"""Reproduce the historical Amazon@10 binary LSTM evaluation.

This command evaluates the two surviving archived fold checkpoints.  Their
presence does not establish how many folds were used in the original training;
the dissertation-era training is reproduced separately with ten folds.  This is
not a future-quarter forecast evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.model_selection import KFold

from PredictionModelPath import AMAZON_REVENUE_10_LSTM_BINARY_CLASS
from classifier.ModelEvaluationHelper import loadModel
from classifier.TweetGroupDataset import TweetGroupDataset
from classifier.transformer.DatasetUtils import createDataloader
from nlpvectors.DataframeSplitter import DataframeSplitter
from nlpvectors.TweetTokenizer import TweetTokenizer
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from tweetpreprocess.EqualClassSampler import EqualClassSampler
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter


def parse_args():
    default_model_dir = Path(AMAZON_REVENUE_10_LSTM_BINARY_CLASS.getModelPath())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_model_dir / "old",
        help="Directory containing dissertation-era checkpoints and split indexes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/old-model-reproduction/amazon-lstm-n10-binary.json"),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def checkpoint_candidates(checkpoint_dir, fold):
    candidates = []
    for name in (
        f"tweetpredict_fold{fold}.ckpt",
        f"tweetpredict_fold{fold}-old.ckpt",
        f"tweetpredict_fold{fold}-v1.ckpt",
    ):
        path = checkpoint_dir / name
        if path.exists() and path.resolve() not in {p.resolve() for p in candidates}:
            candidates.append(path)
    return candidates


def evaluate(model, loader, device):
    true_labels = []
    predictions = []
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            logits = model(inputs.to(device, non_blocking=True))
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            true_labels.extend(labels.numpy().tolist())
    return np.asarray(true_labels), np.asarray(predictions)


def metric_record(true_labels, predictions):
    return {
        "samples": int(len(true_labels)),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "mcc": float(matthews_corrcoef(true_labels, predictions)),
        "true_class_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(true_labels, return_counts=True))
        },
        "predicted_class_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(predictions, return_counts=True))
        },
        "classification_report": classification_report(
            true_labels, predictions, output_dict=True, zero_division=0
        ),
    }


def main():
    args = parse_args()
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = AMAZON_REVENUE_10_LSTM_BINARY_CLASS
    dataframe = pd.read_csv(config.getDataframePath()).fillna("")
    dataframe = EqualClassSampler().getDfWithEqualNumberOfClassSamples(dataframe)
    splitter = DataframeSplitter()
    groups = splitter.getSplitIds(dataframe, config.getTweetGroupSize())
    folds = list(KFold(n_splits=2, shuffle=True, random_state=1337).split(groups))

    word_vectors = KeyedVectors.load_word2vec_format(config.getWordVectorsPath(), binary=False)
    encoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())

    result = {
        "protocol": "evaluation_of_two_surviving_archived_folds",
        "warning": "This label-aware shuffled evaluation is not a future-quarter forecast.",
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": 1337,
        "folds": 2,
        "tweet_group_size": config.getTweetGroupSize(),
        "balanced_tweets": int(len(dataframe)),
        "tweet_groups": int(len(groups)),
        "evaluations": [],
    }

    for fold, (_, test_indexes) in enumerate(folds):
        saved_index_paths = [
            args.checkpoint_dir / f"test_idx_fold{fold}.npy",
            args.checkpoint_dir / f"test_idx_fold{fold}-old.npy",
        ]
        saved_index_path = next((p for p in saved_index_paths if p.exists()), None)
        indexes_match = None
        if saved_index_path is not None:
            indexes_match = bool(np.array_equal(np.load(saved_index_path), test_indexes))
            if not indexes_match:
                raise RuntimeError(
                    f"Saved fold indexes do not match the reconstructed seed-1337 split: {saved_index_path}"
                )

        dataset = TweetGroupDataset(
            dataframe=dataframe,
            splits=groups,
            splitIndexes=test_indexes,
            tokenizer=tokenizer,
            textEncoder=encoder,
        )
        loader = createDataloader(
            dataset,
            args.batch_size,
            args.num_workers,
            encoder.getPADTokenID(),
            shuffle=False,
        )

        for checkpoint in checkpoint_candidates(args.checkpoint_dir, fold):
            print(f"Evaluating fold {fold}: {checkpoint}", flush=True)
            model = loadModel(checkpoint, word_vectors, num_classes=2, device=device)
            true_labels, predictions = evaluate(model, loader, device)
            record = {
                "fold": fold,
                "checkpoint": str(checkpoint),
                "saved_indexes_match_reconstructed_split": indexes_match,
                **metric_record(true_labels, predictions),
            }
            result["evaluations"].append(record)
            print(
                f"fold={fold} accuracy={record['accuracy']:.6f} mcc={record['mcc']:.6f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
