"""Stable quarter text aggregates and change features for future-only evaluation."""

from dataclasses import dataclass, replace
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from classifier.QuarterSequenceDataset import lagged_financial_sequence, prepare_financial_quarters
from classifier.RelevantQuarterTextDataset import RELEVANCE_ANCHORS, RELEVANCE_KEYWORDS


POSITIVE_TERMS = (
    "beat", "beats", "growth", "grow", "growing", "strong", "record", "demand",
    "profit", "higher", "increase", "success", "sold", "delivery", "deliveries",
)
NEGATIVE_TERMS = (
    "miss", "misses", "decline", "weak", "loss", "lower", "decrease", "delay",
    "risk", "fall", "falling", "cut", "problem", "recall", "shortage",
)
UNCERTAINTY_TERMS = (
    "may", "might", "could", "expect", "expected", "estimate", "forecast", "guidance",
    "uncertain", "uncertainty", "risk", "possibly", "potential",
)


def _term_pattern(terms):
    return re.compile(r"\b(?:%s)\b" % "|".join(re.escape(term) for term in terms), re.I)


POSITIVE_PATTERN = _term_pattern(POSITIVE_TERMS)
NEGATIVE_PATTERN = _term_pattern(NEGATIVE_TERMS)
UNCERTAINTY_PATTERN = _term_pattern(UNCERTAINTY_TERMS)
NUMBER_PATTERN = re.compile(r"(?:\$|€|£)?\b\d+(?:[.,]\d+)?(?:%|bn|billion|m|million|k)?\b", re.I)
CURRENCY_OR_PERCENT_PATTERN = re.compile(r"(?:\$|€|£|\b\d+(?:[.,]\d+)?%)")


@dataclass(frozen=True)
class TextDeltaQuarterRecord:
    company: str
    company_index: int
    quarter: str
    label: int
    percent_change: float
    text_sequence: np.ndarray
    financial_sequence: np.ndarray


class TextDeltaQuarterDataset(Dataset):

    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return (
            torch.as_tensor(record.text_sequence, dtype=torch.float32),
            torch.as_tensor(record.financial_sequence, dtype=torch.float32),
            record.company_index,
            int(record.quarter[-1]) - 1,
            record.label,
            float(record.percent_change),
            index,
        )


def _lexical_scores(bodies, keywords):
    pattern = _term_pattern(keywords)
    return bodies.map(lambda body: len(pattern.findall(body)))


def _select_bin_candidates(bin_frame, keywords, maximum, random_state):
    frame = bin_frame.copy()
    frame["lexical_score"] = _lexical_scores(frame["body"], keywords)
    relevant = frame[frame["lexical_score"] > 0]
    if relevant.empty:
        relevant = frame
    if maximum <= 0 or len(relevant) <= maximum:
        return relevant

    # Keep high-density financial language while retaining a uniform sample of the long tail.
    high_count = maximum // 2
    highest = relevant.nlargest(high_count, "lexical_score")
    remaining = relevant.drop(highest.index)
    random_count = min(maximum - len(highest), len(remaining))
    chosen = random_state.choice(len(remaining), size=random_count, replace=False)
    return pd.concat((highest, remaining.iloc[chosen])).sort_values("post_date", kind="stable")


def _fraction_matching(bodies, pattern):
    if not bodies:
        return 0.0
    return float(np.mean([bool(pattern.search(body)) for body in bodies]))


def _bin_statistics(all_bodies, selected, semantic_scores):
    selected_bodies = selected["body"].tolist()
    lexical = selected["lexical_score"].to_numpy(dtype=np.float32)
    return np.asarray([
        np.log1p(len(all_bodies)) / 12.0,
        np.log1p(len(selected_bodies)) / 10.0,
        len(selected_bodies) / max(1.0, len(all_bodies)),
        np.clip(float(lexical.mean()) if len(lexical) else 0.0, 0.0, 8.0) / 8.0,
        float(semantic_scores.mean()),
        float(semantic_scores.max()),
        float(semantic_scores.std()),
        _fraction_matching(selected_bodies, NUMBER_PATTERN),
        _fraction_matching(selected_bodies, CURRENCY_OR_PERCENT_PATTERN),
        _fraction_matching(selected_bodies, POSITIVE_PATTERN),
        _fraction_matching(selected_bodies, NEGATIVE_PATTERN),
        _fraction_matching(selected_bodies, UNCERTAINTY_PATTERN),
    ], dtype=np.float32)


def aggregate_quarter_text_features(tweets, quarters, sentence_model, experiment, bins=8,
                                    max_relevant_per_bin=512, batch_size=512, seed=1337):
    """Aggregate many label-independent finance-relevant tweets into chronological bins."""
    anchors = sentence_model.encode(
        list(RELEVANCE_ANCHORS[experiment]), convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False)
    keywords = RELEVANCE_KEYWORDS[experiment]
    random_state = np.random.RandomState(seed)
    base_features = {}
    for quarter in quarters:
        quarter_frame = tweets[tweets["reporting_quarter"] == quarter].sort_values(
            "post_date", kind="stable")
        if quarter_frame.empty:
            raise ValueError("No tweets found for %s" % quarter)
        frames = []
        counts = []
        for positions in np.array_split(np.arange(len(quarter_frame)), bins):
            bin_frame = quarter_frame.iloc[positions]
            frames.append(_select_bin_candidates(
                bin_frame, keywords, max_relevant_per_bin, random_state))
            counts.append(len(bin_frame))

        selected_bodies = sum((frame["body"].tolist() for frame in frames), [])
        embeddings = sentence_model.encode(
            selected_bodies, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        offset = 0
        quarter_features = []
        for frame, all_count in zip(frames, counts):
            bin_embeddings = embeddings[offset:offset + len(frame)]
            offset += len(frame)
            semantic_scores = (bin_embeddings @ anchors.T).max(axis=1)
            statistics = _bin_statistics([None] * all_count, frame, semantic_scores)
            quarter_features.append(np.concatenate((bin_embeddings.mean(axis=0), statistics)))
        base_features[quarter] = np.asarray(quarter_features, dtype=np.float32)
        print("%s: aggregated %d finance-relevant tweets into %d time bins"
              % (quarter, len(selected_bodies), bins))
    return base_features


def add_quarter_deltas(base_features, quarter):
    """Concatenate current levels, previous-quarter deltas and year-over-year deltas."""
    period = pd.Period(quarter, freq="Q")
    current = base_features[quarter]
    previous = base_features.get(str(period - 1))
    year_ago = base_features.get(str(period - 4))
    previous_delta = np.zeros_like(current) if previous is None else current - previous
    year_delta = np.zeros_like(current) if year_ago is None else current - year_ago
    flags = np.tile(np.asarray([
        float(previous is not None), float(year_ago is not None)
    ], dtype=np.float32), (current.shape[0], 1))
    return np.concatenate((current, previous_delta, year_delta, flags), axis=1).astype(np.float32)


def build_text_delta_records(company, company_index, tweets, financial_frame, base_features,
                             lookback=4):
    financial = prepare_financial_quarters(financial_frame)
    records = []
    for quarter in sorted(base_features):
        target = financial[financial["quarter"] == quarter]
        if target.empty or target["label"].isna().all():
            continue
        records.append(TextDeltaQuarterRecord(
            company=company,
            company_index=company_index,
            quarter=quarter,
            label=int(target["label"].iloc[0]),
            percent_change=float(target["percent_change"].iloc[0]),
            text_sequence=add_quarter_deltas(base_features, quarter),
            financial_sequence=lagged_financial_sequence(financial, quarter, lookback=lookback),
        ))
    return records


def shuffle_text_within_company(records, seed):
    """Break target/text alignment without altering companies, quarters or financial inputs."""
    shuffled = list(records)
    random_state = np.random.RandomState(seed)
    for company in sorted(set(record.company for record in records)):
        indexes = [index for index, record in enumerate(records) if record.company == company]
        if len(indexes) < 2:
            continue
        shift = int(random_state.randint(1, len(indexes)))
        sources = indexes[shift:] + indexes[:shift]
        for target_index, source_index in zip(indexes, sources):
            shuffled[target_index] = replace(
                records[target_index], text_sequence=records[source_index].text_sequence)
    return shuffled
