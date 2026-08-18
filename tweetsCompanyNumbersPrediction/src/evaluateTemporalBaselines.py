'''
Created on 17.08.2026

Reference evaluation that trains no model and reads no tweet text.

The class of a tweet is the change of the financial figure of the reporting period the tweet belongs
to, so the label is constant within a quarter and the dataset contains one label per quarter. This
script measures how far that structure alone carries a prediction, using only the date of a tweet
group:

  majority class   always the most frequent label of the training period
  persistence      always the label of the last training quarter
  seasonal naive   the label of the same calendar quarter of the earlier years

The scores are reported for the strict 80/20 temporal split of
trainNumbersPredictionModelTemporalSplit.py and, in addition, walk forward over every quarter that
has at least one earlier year available. Every text model has to be compared against these numbers,
because a model that does not exceed them has not shown that the tweets contribute anything.

@author: vital
'''
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, matthews_corrcoef

from nlpvectors.DataframeSplitter import DataframeSplitter
from classifier.TemporalSplitDiagnostics import getQuartersOfSplits, printSplitComposition, printBaselines, toDatetime
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_BINARY_CLASS, AMAZON_REVENUE_10_LSTM_MULTI_CLASS, \
    APPLE__EPS_10_LSTM_MULTI_CLASS, TESLA_CAR_SALES_10_LSTM_MULTI_CLASS

MODELS_TO_EVALUATE = [
    ("Amazon revenue, binary", AMAZON_REVENUE_10_LSTM_BINARY_CLASS),
    ("Amazon revenue, 4 classes", AMAZON_REVENUE_10_LSTM_MULTI_CLASS),
    ("Apple EPS, 4 classes", APPLE__EPS_10_LSTM_MULTI_CLASS),
    ("Tesla car sales, 4 classes", TESLA_CAR_SALES_10_LSTM_MULTI_CLASS),
]


def getTemporallySortedSplits(df, predictionModelPath):
    splits = DataframeSplitter().getSplitIds(df, predictionModelPath.getTweetGroupSize())
    df["post_date"] = toDatetime(df["post_date"])
    tweetIdToDate = dict(zip(df["tweet_id"], df["post_date"]))
    tweetIdToClass = dict(zip(df["tweet_id"], df["class"]))
    order = np.argsort([min(tweetIdToDate[tweetId] for tweetId in split) for split in splits])
    splits = [splits[i] for i in order]
    return splits, np.array([tweetIdToClass[split[0]] for split in splits])


def evaluateSeasonalNaiveWalkForward(quartersOfSplits, labelsOfSplits):
    """Predict each quarter with the label of the same calendar quarter of all earlier years."""
    calendarQuarter = np.array([int(quarter[-1]) for quarter in quartersOfSplits])
    year = np.array([int(quarter[:4]) for quarter in quartersOfSplits])
    trueLabels, predictions, correctQuarters, evaluatedQuarters = [], [], 0, 0
    for quarter in sorted(set(quartersOfSplits.tolist())):
        earlierSameQuarter = (year < int(quarter[:4])) & (calendarQuarter == int(quarter[-1]))
        if not earlierSameQuarter.any():
            continue
        prediction = Counter(labelsOfSplits[earlierSameQuarter].tolist()).most_common(1)[0][0]
        isQuarter = quartersOfSplits == quarter
        trueLabels.append(labelsOfSplits[isQuarter])
        predictions.append(np.full(isQuarter.sum(), prediction))
        correctQuarters += prediction == labelsOfSplits[isQuarter][0]
        evaluatedQuarters += 1
    trueLabels, predictions = np.concatenate(trueLabels), np.concatenate(predictions)
    return (accuracy_score(trueLabels, predictions), matthews_corrcoef(trueLabels, predictions),
            correctQuarters, evaluatedQuarters)


if __name__ == "__main__":
    for name, predictionModelPath in MODELS_TO_EVALUATE:
        print("\n" + "=" * 90)
        print(name)
        df = pd.read_csv(predictionModelPath.getDataframePath())
        df.fillna('', inplace=True)
        splits, split_labels = getTemporallySortedSplits(df, predictionModelPath)
        split_quarters = getQuartersOfSplits(df, splits)

        n = len(splits)
        split_point = int(n * 0.8)
        train_val_idx, test_idx = np.arange(0, split_point), np.arange(split_point, n)

        print("Label per quarter:", {quarter: Counter(split_labels[split_quarters == quarter].tolist()).most_common(1)[0][0]
                                     for quarter in sorted(set(split_quarters.tolist()))})
        printSplitComposition(split_quarters, train_val_idx, train_val_idx[:1], test_idx, split_labels)
        printBaselines(split_quarters, split_labels, train_val_idx, test_idx)

        accuracy, mcc, correctQuarters, evaluatedQuarters = evaluateSeasonalNaiveWalkForward(split_quarters, split_labels)
        print("\n=== Seasonal naive, walk forward over every quarter with an earlier year ===")
        print("  accuracy %.3f  MCC %.3f  correctly predicted quarters %d of %d"
              % (accuracy, mcc, correctQuarters, evaluatedQuarters))
