"""Hierarchical tweet-group inputs and leakage-conscious metadata features."""

import math
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


SAFE_METADATA_FEATURE_NAMES = (
    "sentiment_mean",
    "sentiment_std",
    "sentiment_min",
    "sentiment_max",
    "positive_tweet_fraction",
    "negative_tweet_fraction",
    "word_count_mean",
    "word_count_std",
    "character_count_mean",
    "character_count_std",
    "digit_tweet_fraction",
    "currency_tweet_fraction",
    "hashtag_tweet_fraction",
    "mention_tweet_fraction",
    "url_tweet_fraction",
    "question_tweet_fraction",
    "exclamation_tweet_fraction",
    "unique_writer_fraction",
    "quarter_progress_mean",
    "quarter_progress_std",
    "log_group_span_hours",
    "log_quarter_tweet_count",
)


def _fraction(values):
    return float(np.mean(values)) if values else 0.0


def extract_safe_group_metadata(dataframe, group, quarter_tweet_counts, sentiment_analyzer):
    """Features available from tweet content and timestamps by the end of the quarter.

    Likes, retweets and comments are deliberately excluded because the repository does not record
    when those counters were observed.  Their final values could contain post-publication leakage.
    """
    rows = dataframe.loc[list(group.row_indexes)]
    bodies = rows["body"].fillna("").astype(str).tolist()
    sentiments = np.asarray([
        sentiment_analyzer.polarity_scores(body)["compound"] for body in bodies], dtype=np.float32)
    word_counts = np.asarray([len(body.split()) for body in bodies], dtype=np.float32)
    character_counts = np.asarray([len(body) for body in bodies], dtype=np.float32)

    timestamps = pd.to_datetime(rows["post_date"], unit="s", utc=True)
    local = timestamps.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    period = pd.Period(group.quarter, freq="Q")
    quarter_start = period.start_time
    quarter_seconds = max((period.end_time - quarter_start).total_seconds(), 1.0)
    progress = np.asarray([
        (timestamp - quarter_start).total_seconds() / quarter_seconds for timestamp in local],
        dtype=np.float32,
    )
    span_hours = max((local.max() - local.min()).total_seconds() / 3600.0, 0.0)
    writers = rows["writer"].fillna("").astype(str).tolist() if "writer" in rows else []

    return np.asarray([
        sentiments.mean(),
        sentiments.std(),
        sentiments.min(),
        sentiments.max(),
        _fraction((sentiments > 0.05).tolist()),
        _fraction((sentiments < -0.05).tolist()),
        word_counts.mean(),
        word_counts.std(),
        character_counts.mean(),
        character_counts.std(),
        _fraction([bool(re.search(r"\d", body)) for body in bodies]),
        _fraction([bool(re.search(r"[$€£¥]|\b(?:usd|eur|gbp)\b", body, re.I)) for body in bodies]),
        _fraction(["#" in body for body in bodies]),
        _fraction(["@" in body for body in bodies]),
        _fraction([bool(re.search(r"https?://|www\.", body, re.I)) for body in bodies]),
        _fraction(["?" in body for body in bodies]),
        _fraction(["!" in body for body in bodies]),
        len(set(writers)) / max(len(writers), 1),
        progress.mean(),
        progress.std(),
        math.log1p(span_hours),
        math.log1p(quarter_tweet_counts[group.quarter]),
    ], dtype=np.float32)


class MultiViewQuarterGroupDataset(Dataset):
    """Top2Vec token IDs, MiniLM tweet embeddings and safe metadata for each tweet group."""

    def __init__(self, dataframe, groups, tokenizer, text_encoder, sentence_model,
                 max_words_per_tweet=48, sentence_batch_size=512):
        if not groups:
            raise ValueError("At least one tweet group is required")
        group_sizes = {len(group.row_indexes) for group in groups}
        if len(group_sizes) != 1:
            raise ValueError("Hierarchical batches require equally sized tweet groups")

        self.labels = [group.label for group in groups]
        self.quarters = [group.quarter for group in groups]
        self.tweets_per_group = next(iter(group_sizes))
        self.word_ids = []
        all_bodies = []
        unknown = text_encoder.getUNKTokenID()

        for group in groups:
            bodies = dataframe.loc[list(group.row_indexes), "body"].fillna("").astype(str).tolist()
            encoded_group = []
            for body in bodies:
                tokens = tokenizer.tokenize(body)[:max_words_per_tweet]
                encoded = text_encoder.encodeTokens(tokens) or [unknown]
                encoded_group.append(torch.tensor(encoded, dtype=torch.long))
            self.word_ids.append(encoded_group)
            all_bodies.extend(bodies)

        sentence_embeddings = sentence_model.encode(
            all_bodies,
            batch_size=sentence_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.sentence_embeddings = torch.tensor(
            sentence_embeddings.reshape(len(groups), self.tweets_per_group, -1),
            dtype=torch.float32,
        )

        analyzer = SentimentIntensityAnalyzer()
        quarter_tweet_counts = dataframe["reporting_quarter"].value_counts().to_dict()
        metadata = [
            extract_safe_group_metadata(dataframe, group, quarter_tweet_counts, analyzer)
            for group in groups
        ]
        self.metadata = torch.tensor(np.stack(metadata), dtype=torch.float32)
        if self.metadata.shape[1] != len(SAFE_METADATA_FEATURE_NAMES):
            raise AssertionError("Metadata feature names and values are out of sync")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.word_ids[index],
            self.sentence_embeddings[index],
            self.metadata[index],
            self.labels[index],
            self.quarters[index],
        )
