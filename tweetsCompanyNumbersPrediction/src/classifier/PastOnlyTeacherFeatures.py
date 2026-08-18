"""Quarter summaries for a past-only LSTM text teacher.

The hidden coordinates of independently trained company models are not directly comparable.
Consequently this module exports only aligned or rotation-invariant quantities: class
probabilities, votes, confidence/entropy, hidden-vector norms and cosine similarities to class
prototypes learned from past quarters.  A downstream quarter model can therefore combine several
companies without pretending that their raw LSTM axes have the same meaning.
"""

import numpy as np
import pandas as pd


NUM_CLASSES = 4
SUMMARY_FEATURE_NAMES = tuple(
    ["probability_%d_%s" % (class_index, statistic)
     for statistic in ("mean", "std", "q10", "median", "q90")
     for class_index in range(NUM_CLASSES)]
    + ["vote_fraction_%d" % class_index for class_index in range(NUM_CLASSES)]
    + ["entropy_mean", "entropy_std", "confidence_mean", "confidence_std"]
    + ["hidden_norm_%s" % statistic
       for statistic in ("mean", "std", "q10", "median", "q90")]
    + ["prototype_cosine_%d_%s" % (class_index, statistic)
       for statistic in ("mean", "std")
       for class_index in range(NUM_CLASSES)]
    + ["prototype_available_%d" % class_index for class_index in range(NUM_CLASSES)]
)


def _as_two_dimensional(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("%s must be a non-empty two-dimensional array" % name)
    if not np.isfinite(values).all():
        raise ValueError("%s contains non-finite values" % name)
    return values


def compute_class_prototypes(hidden, labels, num_classes=NUM_CLASSES):
    """Return past-only class centroids and flags for classes observed in training."""
    hidden = _as_two_dimensional(hidden, "hidden")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (hidden.shape[0],):
        raise ValueError("labels must contain one value per hidden representation")
    prototypes = np.zeros((num_classes, hidden.shape[1]), dtype=np.float64)
    available = np.zeros(num_classes, dtype=np.float64)
    for class_index in range(num_classes):
        class_hidden = hidden[labels == class_index]
        if len(class_hidden):
            prototypes[class_index] = class_hidden.mean(axis=0)
            available[class_index] = 1.0
    return prototypes.astype(np.float32), available.astype(np.float32)


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def summarize_teacher_outputs(hidden, logits, prototypes, prototype_available):
    """Aggregate all teacher groups of one quarter into a fixed-length feature vector."""
    hidden = _as_two_dimensional(hidden, "hidden")
    logits = _as_two_dimensional(logits, "logits")
    prototypes = np.asarray(prototypes, dtype=np.float64)
    prototype_available = np.asarray(prototype_available, dtype=np.float64)
    if logits.shape != (hidden.shape[0], NUM_CLASSES):
        raise ValueError("logits must have shape (groups, %d)" % NUM_CLASSES)
    if prototypes.shape != (NUM_CLASSES, hidden.shape[1]):
        raise ValueError("prototypes have an incompatible shape")
    if prototype_available.shape != (NUM_CLASSES,):
        raise ValueError("prototype_available must have one flag per class")

    probabilities = _softmax(logits)
    probability_features = np.concatenate([
        reducer(probabilities, axis=0)
        for reducer in (
            np.mean,
            np.std,
            lambda values, axis: np.quantile(values, 0.10, axis=axis),
            lambda values, axis: np.quantile(values, 0.50, axis=axis),
            lambda values, axis: np.quantile(values, 0.90, axis=axis),
        )
    ])
    votes = np.bincount(probabilities.argmax(axis=1), minlength=NUM_CLASSES) / len(probabilities)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    confidence = probabilities.max(axis=1)

    hidden_norms = np.linalg.norm(hidden, axis=1)
    norm_features = np.asarray([
        hidden_norms.mean(),
        hidden_norms.std(),
        np.quantile(hidden_norms, 0.10),
        np.quantile(hidden_norms, 0.50),
        np.quantile(hidden_norms, 0.90),
    ])
    hidden_denominator = np.maximum(hidden_norms[:, None], 1e-12)
    prototype_norms = np.linalg.norm(prototypes, axis=1)
    cosine = hidden @ prototypes.T
    cosine /= hidden_denominator * np.maximum(prototype_norms[None, :], 1e-12)
    cosine *= prototype_available[None, :]
    cosine_features = np.concatenate((cosine.mean(axis=0), cosine.std(axis=0)))

    summary = np.concatenate((
        probability_features,
        votes,
        np.asarray([entropy.mean(), entropy.std(), confidence.mean(), confidence.std()]),
        norm_features,
        cosine_features,
        prototype_available,
    )).astype(np.float32)
    if summary.shape != (len(SUMMARY_FEATURE_NAMES),):
        raise AssertionError("Teacher summary feature names and values are out of sync")
    return summary


def temporal_teacher_features(summary_by_quarter, quarter):
    """Append previous-quarter and year-over-year text changes without future access."""
    period = pd.Period(quarter, freq="Q")
    current = np.asarray(summary_by_quarter[str(period)], dtype=np.float32)
    zeros = np.zeros_like(current)
    previous_key = str(period - 1)
    year_ago_key = str(period - 4)
    previous = np.asarray(summary_by_quarter.get(previous_key, zeros), dtype=np.float32)
    year_ago = np.asarray(summary_by_quarter.get(year_ago_key, zeros), dtype=np.float32)
    return np.concatenate((
        current,
        current - previous if previous_key in summary_by_quarter else zeros,
        current - year_ago if year_ago_key in summary_by_quarter else zeros,
        np.asarray([
            float(previous_key in summary_by_quarter),
            float(year_ago_key in summary_by_quarter),
        ], dtype=np.float32),
    ))


TEMPORAL_FEATURE_NAMES = tuple(
    ["current_%s" % name for name in SUMMARY_FEATURE_NAMES]
    + ["previous_delta_%s" % name for name in SUMMARY_FEATURE_NAMES]
    + ["year_ago_delta_%s" % name for name in SUMMARY_FEATURE_NAMES]
    + ["previous_available", "year_ago_available"]
)
