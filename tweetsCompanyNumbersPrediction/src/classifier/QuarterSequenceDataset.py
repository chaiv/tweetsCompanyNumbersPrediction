"""Quarter-level records with current text and strictly lagged financial history."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from classifier.QuarterAlignedDataset import reporting_quarters


FINANCIAL_FEATURE_NAMES = (
    "log_level_relative_to_latest_lag",
    "percent_change_clipped",
    "year_over_year_log_change",
    "available",
    "calendar_q1",
    "calendar_q2",
    "calendar_q3",
    "calendar_q4",
    "lag_class_0",
    "lag_class_1",
    "lag_class_2",
    "lag_class_3",
)


def percent_change_class(percent_change):
    """Map a quarterly percentage change to the repository's four classes."""
    if pd.isna(percent_change):
        raise ValueError("A target quarter cannot have a missing percent_change")
    if percent_change < 0:
        return 0
    if percent_change <= 15:
        return 1
    if percent_change <= 30:
        return 2
    return 3


def prepare_financial_quarters(dataframe):
    """Parse and validate one row per reporting quarter."""
    frame = dataframe.copy()
    frame["quarter"] = pd.to_datetime(
        frame["from_date"], dayfirst=True).dt.to_period("Q").astype(str)
    if frame["quarter"].duplicated().any():
        duplicates = frame.loc[frame["quarter"].duplicated(), "quarter"].tolist()
        raise ValueError("Duplicate financial quarters: %s" % duplicates)
    frame["label"] = frame["percent_change"].map(
        lambda value: np.nan if pd.isna(value) else percent_change_class(value))
    frame.sort_values("quarter", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def lagged_financial_sequence(financial_quarters, target_quarter, lookback=4):
    """Build t-lookback..t-1 features without reading the target quarter's value.

    Raw levels are made company-scale independent by expressing every lag relative to the latest
    available lag.  Missing early history is represented explicitly by the ``available`` feature.
    """
    target = pd.Period(target_quarter, freq="Q")
    indexed = financial_quarters.set_index("quarter")
    prior_quarters = [target - lag for lag in range(lookback, 0, -1)]
    available_values = []
    for quarter in prior_quarters:
        key = str(quarter)
        if key in indexed.index:
            available_values.append(float(indexed.loc[key, "value"]))
    latest_value = available_values[-1] if available_values else 1.0
    latest_value = max(abs(latest_value), 1e-12)

    rows = []
    for quarter in prior_quarters:
        key = str(quarter)
        if key not in indexed.index:
            rows.append(np.zeros(len(FINANCIAL_FEATURE_NAMES), dtype=np.float32))
            continue
        row = indexed.loc[key]
        value = max(abs(float(row["value"])), 1e-12)
        percent_change = 0.0 if pd.isna(row["percent_change"]) else float(row["percent_change"])
        year_ago_key = str(quarter - 4)
        if year_ago_key in indexed.index:
            year_ago_value = max(abs(float(indexed.loc[year_ago_key, "value"])), 1e-12)
            yoy_change = np.log(value / year_ago_value)
        else:
            yoy_change = 0.0
        calendar = np.zeros(4, dtype=np.float32)
        calendar[quarter.quarter - 1] = 1.0
        lag_class = np.zeros(4, dtype=np.float32)
        if not pd.isna(row["percent_change"]):
            lag_class[percent_change_class(percent_change)] = 1.0
        rows.append(np.concatenate((np.asarray([
            np.clip(np.log(value / latest_value), -3.0, 3.0),
            np.clip(percent_change / 100.0, -2.0, 3.0),
            np.clip(yoy_change, -3.0, 3.0),
            1.0,
        ], dtype=np.float32), calendar, lag_class)))
    return np.asarray(rows, dtype=np.float32)


def select_text_bags(dataframe, quarter, bins=8, tweets_per_bin=8, variants=8, seed=1337,
                     date_column="post_date", text_column="body"):
    """Select deterministic chronological tweet bags spread across a whole quarter."""
    if "reporting_quarter" not in dataframe:
        frame = dataframe.copy()
        frame["reporting_quarter"] = reporting_quarters(frame[date_column])
    else:
        frame = dataframe
    quarter_frame = frame[frame["reporting_quarter"] == quarter].sort_values(
        date_column, kind="stable")
    if quarter_frame.empty:
        raise ValueError("No tweets available for quarter %s" % quarter)

    positions = np.arange(len(quarter_frame))
    bin_indexes = np.array_split(positions, bins)
    random = np.random.RandomState(seed)
    result = []
    texts = quarter_frame[text_column].fillna("").astype(str).tolist()
    for _ in range(variants):
        variant = []
        for indexes in bin_indexes:
            if len(indexes) == 0:
                variant.append([""] * tweets_per_bin)
                continue
            chosen = random.choice(
                indexes, size=tweets_per_bin, replace=len(indexes) < tweets_per_bin)
            variant.append([texts[index] for index in chosen])
        result.append(variant)
    return result


@dataclass(frozen=True)
class QuarterSequenceRecord:
    company: str
    company_index: int
    quarter: str
    label: int
    text_sequences: np.ndarray
    financial_sequence: np.ndarray


class QuarterSequenceDataset(Dataset):
    """Flatten text-bag variants while retaining the independent quarter identity."""

    def __init__(self, records):
        self.records = list(records)
        self.index = [
            (record_index, variant_index)
            for record_index, record in enumerate(self.records)
            for variant_index in range(record.text_sequences.shape[0])
        ]
        self.labels = [record.label for record in self.records]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        record_index, variant_index = self.index[index]
        record = self.records[record_index]
        return (
            torch.as_tensor(record.text_sequences[variant_index], dtype=torch.float32),
            torch.as_tensor(record.financial_sequence, dtype=torch.float32),
            record.company_index,
            int(record.quarter[-1]) - 1,
            record.label,
            record_index,
        )
