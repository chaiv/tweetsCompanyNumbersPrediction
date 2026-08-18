'''
Created on 17.08.2026

Diagnostics and reference baselines for temporally ordered evaluations.

The class label of a tweet is the change of the financial figure of the reporting period the tweet
belongs to, so every tweet of a quarter carries the same label and the whole dataset contains only
one label per quarter. Two consequences follow, and both are made visible here:

  1. A split that places tweet groups of the same quarter into train and test can be solved by
     recognizing the period a text comes from, without any predictive content. getQuarterOverlap
     reports how large that overlap is.
  2. A strictly temporal split decides only as many independent cases as there are quarters in the
     test period. calculateBaselines therefore reports what is reachable without reading the text:
     the majority class, the label of the last training quarter (persistence) and the label of the
     same calendar quarter of earlier years (seasonal naive). A text model is only informative if
     it beats these.

@author: vital
'''
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef


def toDatetime(dateColumn, timezone="Europe/Berlin"):
    """Convert epoch seconds to the local time used for financial-period boundaries.

    DateTSPConverter created boundaries from naive local datetimes.  Epoch seconds are UTC, so the
    timezone conversion is required before deriving a reporting quarter; otherwise tweets around
    midnight at a quarter boundary are assigned to the adjacent calendar quarter.
    """
    if pd.api.types.is_numeric_dtype(dateColumn):
        datetimes = pd.to_datetime(dateColumn, unit="s", utc=True)
        if isinstance(datetimes, pd.Series):
            return datetimes.dt.tz_convert(timezone).dt.tz_localize(None)
        return datetimes.tz_convert(timezone).tz_localize(None)
    return pd.to_datetime(dateColumn)


def getQuartersOfSplits(df, splits, idColumnName="tweet_id", dateColumnName="post_date"):
    """Reporting quarter of each tweet group, taken from its earliest tweet."""
    tweetIdToDate = dict(zip(df[idColumnName], toDatetime(df[dateColumnName])))
    return np.array([str(min(tweetIdToDate[tweetId] for tweetId in split).to_period("Q")) for split in splits])


def getQuarterOverlap(quartersOfSplits, trainIndexes, testIndexes):
    """Share of test groups whose quarter also occurs in the training data."""
    trainQuarters = set(quartersOfSplits[trainIndexes].tolist())
    testQuarters = quartersOfSplits[testIndexes]
    return float(np.isin(testQuarters, list(trainQuarters)).mean())


def printSplitComposition(quartersOfSplits, trainIndexes, valIndexes, testIndexes, labelsOfSplits):
    trainQuarters = sorted(set(quartersOfSplits[trainIndexes].tolist()))
    valQuarters = sorted(set(quartersOfSplits[valIndexes].tolist()))
    testQuarters = sorted(set(quartersOfSplits[testIndexes].tolist()))
    overlap = getQuarterOverlap(quartersOfSplits, trainIndexes, testIndexes)
    print("Train quarters", trainQuarters)
    print("Val quarters  ", valQuarters)
    print("Test quarters ", testQuarters)
    print("Share of test groups whose quarter also occurs in train: %.4f" % overlap)
    print("Distinct labels in the test period: %d over %d quarters, so the test set decides %d "
          "independent cases regardless of how many tweet groups it contains."
          % (len(set(labelsOfSplits[testIndexes].tolist())), len(testQuarters), len(testQuarters)))
    if overlap > 0:
        print("WARNING: test quarters also occur in training. Because the label is constant within a "
              "quarter, the model can reach a high score by recognizing the period of a text.")


def calculateBaselines(quartersOfSplits, labelsOfSplits, trainIndexes, testIndexes):
    """Reference scores reachable without reading the tweets.

    majority   - always the most frequent training label
    persistence- always the label of the last training quarter
    seasonal   - the majority label of the same calendar quarter in the training period, which uses
                 only the date of the test group, a value that is known in a real forecast

    Returns a dict of name -> (accuracy, mcc).
    """
    trainLabels = labelsOfSplits[trainIndexes]
    testLabels = labelsOfSplits[testIndexes]
    calendarQuarter = np.array([int(quarter[-1]) for quarter in quartersOfSplits])

    baselines = {}
    majority = Counter(trainLabels.tolist()).most_common(1)[0][0]
    baselines["majority class"] = np.full(len(testLabels), majority)

    lastTrainQuarter = sorted(set(quartersOfSplits[trainIndexes].tolist()))[-1]
    lastLabel = Counter(labelsOfSplits[quartersOfSplits == lastTrainQuarter].tolist()).most_common(1)[0][0]
    baselines["persistence (label of %s)" % lastTrainQuarter] = np.full(len(testLabels), lastLabel)

    seasonalMap = {quarter: Counter(trainLabels[calendarQuarter[trainIndexes] == quarter].tolist()).most_common(1)[0][0]
                   for quarter in sorted(set(calendarQuarter[trainIndexes].tolist()))}
    baselines["seasonal naive %s" % seasonalMap] = np.array(
        [seasonalMap[quarter] for quarter in calendarQuarter[testIndexes]])

    return {name: (accuracy_score(testLabels, prediction), matthews_corrcoef(testLabels, prediction))
            for name, prediction in baselines.items()}


def printBaselines(quartersOfSplits, labelsOfSplits, trainIndexes, testIndexes):
    print("\n=== Reference baselines that do not read the tweets ===")
    for name, (accuracy, mcc) in calculateBaselines(quartersOfSplits, labelsOfSplits, trainIndexes, testIndexes).items():
        print("  %-46s accuracy %.3f  MCC %.3f" % (name, accuracy, mcc))
    print("  A text model is informative only where it exceeds these values.")
