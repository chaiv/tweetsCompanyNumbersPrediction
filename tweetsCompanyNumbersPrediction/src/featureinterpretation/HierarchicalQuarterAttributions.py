"""Token, tweet and topic attributions for the explainable quarter text residual."""

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients


class _ScaledTextContribution(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, word_ids):
        scale = torch.nn.functional.softplus(self.model.text_residual_scale)
        return scale * self.model.text_logits(word_ids)


class HierarchicalQuarterAttributions:
    """Integrated Gradients for the actual scaled text contribution to fusion logits."""

    def __init__(self, model):
        self.model = model
        self.wrapper = _ScaledTextContribution(model)
        self.integrated_gradients = LayerIntegratedGradients(
            self.wrapper, model.word_embedding)

    def attribute(self, word_ids, target, n_steps=64, internal_batch_size=None):
        self.model.eval()
        baseline = torch.full_like(word_ids, self.model.pad_token_idx)
        attributions, delta = self.integrated_gradients.attribute(
            word_ids,
            baselines=baseline,
            target=target,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )
        raw = attributions.sum(dim=-1)
        raw = raw.masked_fill(word_ids.eq(self.model.pad_token_idx), 0.0)
        tweet_signed = raw.sum(dim=2)
        tweet_absolute = raw.abs().sum(dim=2)
        return {
            "token_signed": raw.detach().cpu(),
            "token_absolute": raw.abs().detach().cpu(),
            "tweet_signed": tweet_signed.detach().cpu(),
            "tweet_absolute": tweet_absolute.detach().cpu(),
            "convergence_delta": delta.detach().cpu(),
        }


def token_attributions_to_dataframe(word_ids, tweet_ids, attributions, index_to_key,
                                    quarter, target_class):
    """Keep raw comparable values; normalization is intentionally left to visualization."""
    word_ids = torch.as_tensor(word_ids).cpu()
    tweet_ids = torch.as_tensor(tweet_ids).cpu()
    rows = []
    for batch_index in range(word_ids.shape[0]):
        for tweet_index in range(word_ids.shape[1]):
            tweet_id = int(tweet_ids[batch_index, tweet_index])
            for token_position in range(word_ids.shape[2]):
                word_id = int(word_ids[batch_index, tweet_index, token_position])
                signed = float(attributions["token_signed"][
                    batch_index, tweet_index, token_position])
                absolute = float(attributions["token_absolute"][
                    batch_index, tweet_index, token_position])
                if absolute == 0.0:
                    continue
                rows.append({
                    "quarter": quarter,
                    "target_class": int(target_class),
                    "tweet_id": tweet_id,
                    "tweet_position": tweet_index,
                    "token_position": token_position,
                    "token_id": word_id,
                    "token": index_to_key[word_id],
                    "token_attribution": signed,
                    "token_attribution_abs": absolute,
                    "tweet_attribution": float(attributions["tweet_signed"][
                        batch_index, tweet_index]),
                    "tweet_attribution_abs": float(attributions["tweet_absolute"][
                        batch_index, tweet_index]),
                    "convergence_delta": float(attributions["convergence_delta"][batch_index]),
                })
    return pd.DataFrame(rows)


def aggregate_topic_attributions(attribution_frame, topic_mapping,
                                 tweet_id_column="tweet_id", topic_column="topic_id"):
    """Aggregate signed/absolute contributions after mapping held-out tweets to topics."""
    mapping = topic_mapping[[tweet_id_column, topic_column]].drop_duplicates(tweet_id_column)
    merged = attribution_frame.merge(mapping, on=tweet_id_column, how="left")
    merged[topic_column] = merged[topic_column].fillna(-1).astype(int)
    return merged.groupby(["quarter", "target_class", topic_column], as_index=False).agg(
        token_attribution=("token_attribution", "sum"),
        token_attribution_abs=("token_attribution_abs", "sum"),
        tweet_count=(tweet_id_column, "nunique"),
        token_count=("token", "size"),
    )
