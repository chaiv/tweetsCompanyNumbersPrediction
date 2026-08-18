"""Label-independent relevance selection and token-preserving quarter bags."""

from dataclasses import dataclass, replace
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from classifier.QuarterAlignedDataset import reporting_quarters
from classifier.QuarterSequenceDataset import lagged_financial_sequence, prepare_financial_quarters


RELEVANCE_ANCHORS = {
    "amazon-revenue-4class": (
        "Amazon quarterly revenue sales growth earnings guidance",
        "AWS revenue cloud sales operating margin",
        "Prime retail demand holiday sales profit",
    ),
    "apple-eps": (
        "Apple quarterly earnings per share EPS guidance",
        "iPhone sales revenue gross margin profit",
        "Apple earnings results demand and product sales",
    ),
    "tesla-sales": (
        "Tesla quarterly vehicle deliveries and automotive sales",
        "Tesla production orders factory output Model 3",
        "Tesla delivery numbers demand vehicle production",
    ),
}

RELEVANCE_KEYWORDS = {
    "amazon-revenue-4class": (
        "revenue", "sales", "earnings", "guidance", "profit", "margin", "aws",
        "prime", "retail", "quarter", "q1", "q2", "q3", "q4",
    ),
    "apple-eps": (
        "eps", "earnings", "revenue", "sales", "guidance", "profit", "margin",
        "iphone", "demand", "quarter", "q1", "q2", "q3", "q4",
    ),
    "tesla-sales": (
        "deliveries", "delivery", "delivered", "production", "sales", "orders",
        "factory", "vehicle", "vehicles", "model 3", "model s", "model x", "quarter",
        "q1", "q2", "q3", "q4",
    ),
}


@dataclass(frozen=True)
class RelevantQuarterRecord:
    quarter: str
    label: int
    word_ids: np.ndarray
    tweet_ids: np.ndarray
    financial_sequence: np.ndarray


def _lexical_scores(bodies, keywords):
    pattern = re.compile("|".join(re.escape(keyword) for keyword in keywords), re.IGNORECASE)
    return bodies.map(lambda value: len(pattern.findall(value)))


def _candidate_subset(quarter_frame, keywords, max_candidates, bins, random):
    frame = quarter_frame.sort_values("post_date", kind="stable").copy()
    frame["lexical_score"] = _lexical_scores(frame["body"], keywords)
    matching = frame[frame["lexical_score"] > 0]
    if matching.empty:
        matching = frame
    if len(matching) <= max_candidates:
        return matching

    # Preserve the whole quarter rather than allowing a large event at the end to dominate.
    per_bin = max(1, max_candidates // bins)
    selected = []
    for indexes in np.array_split(np.arange(len(matching)), bins):
        bin_frame = matching.iloc[indexes]
        high_score_count = min(len(bin_frame), per_bin // 2)
        highest = bin_frame.nlargest(high_score_count, "lexical_score")
        remaining = bin_frame.drop(highest.index)
        random_count = min(len(remaining), per_bin - high_score_count)
        if random_count:
            chosen = random.choice(len(remaining), size=random_count, replace=False)
            selected.append(remaining.iloc[chosen])
        selected.append(highest)
    return pd.concat(selected).sort_values("post_date", kind="stable").head(max_candidates)


def select_relevant_tweet_pools(frame, quarters, sentence_model, experiment, bins=8,
                                pool_per_bin=48, max_candidates_per_quarter=4096,
                                sentence_batch_size=512, seed=1337):
    """Return relevant tweet-row pools per quarter and time bin without looking at labels."""
    if experiment not in RELEVANCE_ANCHORS:
        raise ValueError("No relevance anchors configured for %s" % experiment)
    anchors = sentence_model.encode(
        list(RELEVANCE_ANCHORS[experiment]), convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False)
    random = np.random.RandomState(seed)
    pools = {}
    for quarter in quarters:
        quarter_frame = frame[frame["reporting_quarter"] == quarter]
        if quarter_frame.empty:
            raise ValueError("No tweets found for quarter %s" % quarter)
        candidates = _candidate_subset(
            quarter_frame, RELEVANCE_KEYWORDS[experiment],
            max_candidates_per_quarter, bins, random)
        embeddings = sentence_model.encode(
            candidates["body"].tolist(), batch_size=sentence_batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        candidates = candidates.copy()
        candidates["semantic_score"] = (embeddings @ anchors.T).max(axis=1)
        candidates["relevance_score"] = (
            candidates["semantic_score"] + 0.04 * candidates["lexical_score"].clip(upper=5))
        candidates.sort_values("post_date", kind="stable", inplace=True)
        quarter_pools = []
        for indexes in np.array_split(np.arange(len(candidates)), bins):
            bin_frame = candidates.iloc[indexes]
            if bin_frame.empty:
                bin_frame = candidates
            quarter_pools.append(
                bin_frame.nlargest(min(pool_per_bin, len(bin_frame)), "relevance_score").index.tolist())
        pools[quarter] = quarter_pools
        print("%s: %d keyword candidates -> %d relevant tweet pool"
              % (quarter, len(candidates), sum(len(values) for values in quarter_pools)))
    return pools


def build_relevant_quarter_records(frame, financial_frame, pools, tokenizer, encoder,
                                   variants=8, tweets_per_bin=4, max_words=40, seed=1337):
    financial = prepare_financial_quarters(financial_frame)
    random = np.random.RandomState(seed)
    records = []
    for quarter in sorted(pools):
        financial_row = financial[financial["quarter"] == quarter]
        if financial_row.empty or financial_row["label"].isna().all():
            continue
        label = int(financial_row["label"].iloc[0])
        quarter_words, quarter_tweet_ids = [], []
        for _ in range(variants):
            selected_indexes = []
            for bin_pool in pools[quarter]:
                selected_indexes.extend(random.choice(
                    bin_pool, size=tweets_per_bin,
                    replace=len(bin_pool) < tweets_per_bin).tolist())
            selected = frame.loc[selected_indexes].sort_values("post_date", kind="stable")
            word_ids = []
            for body in selected["body"].tolist():
                tokens = tokenizer.tokenize(body)[:max_words]
                encoded = encoder.encodeTokens(tokens) or [encoder.getUNKTokenID()]
                encoded = encoded[:max_words] + [encoder.getPADTokenID()] * (max_words - len(encoded))
                word_ids.append(encoded)
            quarter_words.append(word_ids)
            quarter_tweet_ids.append(selected["tweet_id"].astype(np.int64).tolist())
        records.append(RelevantQuarterRecord(
            quarter=quarter,
            label=label,
            word_ids=np.asarray(quarter_words, dtype=np.int64),
            tweet_ids=np.asarray(quarter_tweet_ids, dtype=np.int64),
            financial_sequence=lagged_financial_sequence(financial, quarter, lookback=4),
        ))
    return records


def shuffle_record_text(records, seed):
    """Cyclic negative control that preserves labels/financial inputs but breaks quarter text."""
    if len(records) < 2:
        return list(records)
    random = np.random.RandomState(seed)
    shift = int(random.randint(1, len(records)))
    sources = list(records[shift:]) + list(records[:shift])
    return [replace(target, word_ids=source.word_ids, tweet_ids=source.tweet_ids)
            for target, source in zip(records, sources)]


class RelevantQuarterDataset(Dataset):

    def __init__(self, records):
        self.records = list(records)
        self.index = [(record_index, variant_index)
                      for record_index, record in enumerate(self.records)
                      for variant_index in range(record.word_ids.shape[0])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        record_index, variant_index = self.index[index]
        record = self.records[record_index]
        return (
            torch.as_tensor(record.word_ids[variant_index], dtype=torch.long),
            torch.as_tensor(record.financial_sequence, dtype=torch.float32),
            int(record.quarter[-1]) - 1,
            record.label,
            record_index,
        )
