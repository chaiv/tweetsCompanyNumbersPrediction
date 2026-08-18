"""Leakage-conscious current-quarter author, volume and timing features."""

import numpy as np
import pandas as pd

from classifier.QuarterAlignedDataset import reporting_quarters


QUARTER_METADATA_FEATURE_NAMES = (
    "log_tweet_count",
    "log_unique_authors",
    "normalized_author_entropy",
    "author_hhi",
    "top_author_fraction",
    "first_month_fraction",
    "second_month_fraction",
    "third_month_fraction",
    "weekend_fraction",
    "url_fraction",
    "cashtag_fraction",
    "number_fraction",
    "hashtag_fraction",
    "mention_fraction",
    "mean_log_text_length",
)


def _fraction(series):
    return float(series.mean()) if len(series) else 0.0


def build_quarter_metadata(dataframe, date_column="post_date", text_column="body",
                           author_column="writer"):
    """Create one target-independent metadata vector per current reporting quarter."""
    required = {date_column, text_column, author_column}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError("Missing metadata columns: %s" % sorted(missing))
    frame = dataframe[[date_column, text_column, author_column]].copy()
    frame["reporting_quarter"] = reporting_quarters(frame[date_column])
    frame["local_datetime"] = pd.to_datetime(
        frame[date_column], unit="s", utc=True).dt.tz_convert(
            "Europe/Berlin").dt.tz_localize(None)
    frame[text_column] = frame[text_column].fillna("").astype(str)
    frame[author_column] = frame[author_column].fillna("<missing>").astype(str)

    result = {}
    for quarter, values in frame.groupby("reporting_quarter", sort=True):
        count = len(values)
        author_counts = values[author_column].value_counts(dropna=False).to_numpy(
            dtype=np.float64)
        author_probabilities = author_counts / max(float(author_counts.sum()), 1.0)
        unique_authors = len(author_counts)
        entropy = -float(np.sum(
            author_probabilities * np.log(np.maximum(author_probabilities, 1e-12))))
        normalized_entropy = entropy / np.log(unique_authors) if unique_authors > 1 else 0.0
        month_position = (values["local_datetime"].dt.month.to_numpy() - 1) % 3
        bodies = values[text_column]
        metadata = np.asarray([
            np.log1p(count),
            np.log1p(unique_authors),
            normalized_entropy,
            float(np.sum(author_probabilities ** 2)),
            float(author_probabilities.max()) if unique_authors else 0.0,
            float(np.mean(month_position == 0)),
            float(np.mean(month_position == 1)),
            float(np.mean(month_position == 2)),
            _fraction(values["local_datetime"].dt.dayofweek >= 5),
            _fraction(bodies.str.contains(r"https?://|www\.", case=False, regex=True)),
            _fraction(bodies.str.contains(r"\$[A-Za-z]{1,6}\b", regex=True)),
            _fraction(bodies.str.contains(r"\d", regex=True)),
            _fraction(bodies.str.contains("#", regex=False)),
            _fraction(bodies.str.contains("@", regex=False)),
            float(np.log1p(bodies.str.len()).mean()),
        ], dtype=np.float32)
        result[str(quarter)] = metadata
    return result


def temporal_metadata_features(metadata_by_quarter, quarter):
    """Return current metadata plus past-only quarter and year-over-year changes."""
    period = pd.Period(quarter, freq="Q")
    current = np.asarray(metadata_by_quarter[str(period)], dtype=np.float32)
    zeros = np.zeros_like(current)
    previous_key, year_ago_key = str(period - 1), str(period - 4)
    previous = np.asarray(metadata_by_quarter.get(previous_key, zeros), dtype=np.float32)
    year_ago = np.asarray(metadata_by_quarter.get(year_ago_key, zeros), dtype=np.float32)
    return np.concatenate((
        current,
        current - previous if previous_key in metadata_by_quarter else zeros,
        current - year_ago if year_ago_key in metadata_by_quarter else zeros,
        np.asarray([
            float(previous_key in metadata_by_quarter),
            float(year_ago_key in metadata_by_quarter),
        ], dtype=np.float32),
    ))


TEMPORAL_METADATA_FEATURE_NAMES = tuple(
    ["current_%s" % name for name in QUARTER_METADATA_FEATURE_NAMES]
    + ["previous_delta_%s" % name for name in QUARTER_METADATA_FEATURE_NAMES]
    + ["year_ago_delta_%s" % name for name in QUARTER_METADATA_FEATURE_NAMES]
    + ["previous_available", "year_ago_available"]
)
