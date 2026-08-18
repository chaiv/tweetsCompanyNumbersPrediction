"""Quarter-aligned tweet groups for leakage-resistant temporal evaluation."""

from dataclasses import dataclass
from itertools import chain

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_REPORTING_TIMEZONE = "Europe/Berlin"


def reporting_quarters(epoch_seconds, timezone=DEFAULT_REPORTING_TIMEZONE):
    """Convert UTC epoch seconds to the local quarters used to construct the labels.

    DateTSPConverter created the financial-period boundaries from naive local datetimes.  Treating
    tweet timestamps as naive UTC moves tweets around midnight at a quarter boundary into the wrong
    period.  Converting to the local timezone before dropping timezone information reproduces the
    reporting periods exactly for the repository's data.
    """
    datetimes = pd.to_datetime(epoch_seconds, unit="s", utc=True)
    if isinstance(datetimes, pd.Series):
        local = datetimes.dt.tz_convert(timezone).dt.tz_localize(None)
        return local.dt.to_period("Q").astype(str)
    local = datetimes.tz_convert(timezone).tz_localize(None)
    return local.to_period("Q").astype(str)


@dataclass(frozen=True)
class QuarterGroup:
    row_indexes: tuple
    quarter: str
    label: int


def build_quarter_groups(dataframe, group_size, date_column="post_date", label_column="class",
                         timezone=DEFAULT_REPORTING_TIMEZONE, drop_remainder=True):
    """Return a time-sorted frame and groups that never cross a reporting-quarter boundary."""
    frame = dataframe.copy()
    frame["reporting_quarter"] = reporting_quarters(frame[date_column], timezone=timezone)
    frame.sort_values(date_column, kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    groups = []
    for quarter, quarter_frame in frame.groupby("reporting_quarter", sort=True):
        labels = sorted(int(value) for value in quarter_frame[label_column].unique())
        if len(labels) != 1:
            raise ValueError(
                "Reporting quarter %s contains multiple labels %s; check timezone or target join"
                % (quarter, labels))
        indexes = quarter_frame.index.to_numpy()
        stop = len(indexes) - (len(indexes) % group_size) if drop_remainder else len(indexes)
        for start in range(0, stop, group_size):
            row_indexes = indexes[start:min(start + group_size, len(indexes))]
            if len(row_indexes) == group_size or not drop_remainder:
                groups.append(QuarterGroup(tuple(row_indexes.tolist()), str(quarter), labels[0]))
    return frame, groups


def select_balanced_quarter_groups(groups, quarters, max_groups_per_quarter, seed=1337):
    """Sample the same maximum number of groups from every requested quarter."""
    random = np.random.RandomState(seed)
    selected = []
    for quarter in sorted(quarters):
        candidates = [group for group in groups if group.quarter == quarter]
        if not candidates:
            raise ValueError("No tweet groups found for quarter %s" % quarter)
        count = min(len(candidates), max_groups_per_quarter)
        indexes = random.choice(len(candidates), size=count, replace=False)
        selected.extend(candidates[index] for index in sorted(indexes.tolist()))
    return selected


class EncodedQuarterGroupDataset(Dataset):
    """Pre-tokenized groups with a per-tweet budget so every tweet remains represented."""

    def __init__(self, dataframe, groups, tokenizer, text_encoder, max_tokens=384,
                 text_column="body"):
        self.quarters = [group.quarter for group in groups]
        self.labels = [group.label for group in groups]
        self.encoded_groups = []
        separator = text_encoder.getSEPTokenID()
        unknown = text_encoder.getUNKTokenID()

        for group in groups:
            bodies = dataframe.loc[list(group.row_indexes), text_column].fillna("").astype(str).tolist()
            token_budget = max(1, (max_tokens - max(0, len(bodies) - 1)) // max(1, len(bodies)))
            encoded_tweets = []
            for body in bodies:
                tokens = tokenizer.tokenize(body)[:token_budget]
                encoded_tweets.append(text_encoder.encodeTokens(tokens))
            encoded = list(chain.from_iterable(
                values + ([separator] if index < len(encoded_tweets) - 1 else [])
                for index, values in enumerate(encoded_tweets)))
            if not encoded:
                encoded = [unknown]
            self.encoded_groups.append(torch.tensor(encoded[:max_tokens], dtype=torch.long))

    def __len__(self):
        return len(self.encoded_groups)

    def __getitem__(self, index):
        return self.encoded_groups[index], self.labels[index], self.quarters[index]
