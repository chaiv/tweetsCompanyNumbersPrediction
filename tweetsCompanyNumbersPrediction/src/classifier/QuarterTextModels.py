"""Compact text classifiers for quarter-aligned financial nowcasting experiments.

The original LSTM consumes right-padded sequences as if padding were text and then uses the hidden
state after the padding.  The models in this module are padding invariant by construction.  The
BiLSTM packs the real tokens, pools them with masked attention and retains a direct residual path
from the mean pretrained embedding.  A mean-embedding MLP is provided as a non-recurrent ablation.
"""

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _masked_mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


class MeanEmbeddingClassifier(torch.nn.Module):
    """Small regularized classifier over the mean of pretrained word embeddings."""

    def __init__(self, word_vectors, pad_token_idx, num_classes, hidden_size=128,
                 dropout=0.35, freeze_embeddings=True):
        super().__init__()
        weights = torch.as_tensor(word_vectors, dtype=torch.float32)
        self.embedding = torch.nn.Embedding.from_pretrained(
            weights, freeze=freeze_embeddings, padding_idx=pad_token_idx)
        self.pad_token_idx = pad_token_idx
        embedding_size = weights.shape[1]
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embedding_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, token_ids, calendar_quarters=None):
        mask = token_ids.ne(self.pad_token_idx)
        pooled = _masked_mean(self.embedding(token_ids), mask)
        return self.classifier(pooled)


class PackedAttentionLSTMClassifier(torch.nn.Module):
    """Padding-safe BiLSTM with attention and a mean-embedding residual connection."""

    def __init__(self, word_vectors, pad_token_idx, num_classes, hidden_size=96,
                 dropout=0.35, freeze_embeddings=True):
        super().__init__()
        weights = torch.as_tensor(word_vectors, dtype=torch.float32)
        self.embedding = torch.nn.Embedding.from_pretrained(
            weights, freeze=freeze_embeddings, padding_idx=pad_token_idx)
        self.pad_token_idx = pad_token_idx
        embedding_size = weights.shape[1]
        recurrent_size = hidden_size * 2

        self.input_norm = torch.nn.LayerNorm(embedding_size)
        self.input_dropout = torch.nn.Dropout(dropout * 0.5)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(recurrent_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False),
        )
        self.embedding_residual = torch.nn.Linear(embedding_size, recurrent_size)
        self.residual_gate = torch.nn.Linear(recurrent_size * 2, recurrent_size)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(recurrent_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(recurrent_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, token_ids, calendar_quarters=None):
        mask = token_ids.ne(self.pad_token_idx)
        lengths = mask.sum(dim=1).clamp(min=1)
        embeddings = self.embedding(token_ids)
        recurrent_input = self.input_dropout(self.input_norm(embeddings))

        packed = pack_padded_sequence(
            recurrent_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=token_ids.shape[1])

        attention_scores = self.attention(outputs).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, torch.finfo(outputs.dtype).min)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attended = (outputs * attention_weights.unsqueeze(-1)).sum(dim=1)

        residual = self.embedding_residual(_masked_mean(embeddings, mask))
        gate = torch.sigmoid(self.residual_gate(torch.cat([attended, residual], dim=1)))
        pooled = gate * attended + (1.0 - gate) * residual
        return self.classifier(pooled)


class SeasonalResidualEmbeddingClassifier(torch.nn.Module):
    """Known seasonal prior plus a trainable residual derived from word embeddings.

    Calendar quarter is known at prediction time and is an unusually strong baseline for these
    financial series.  Keeping it in a separate additive branch makes the comparison honest: the
    residual branch has to correct a seasonal prediction, rather than forcing the text encoder to
    rediscover the calendar from period-specific vocabulary.
    """

    def __init__(self, word_vectors, pad_token_idx, num_classes, seasonal_log_prior,
                 hidden_size=128, dropout=0.35, freeze_embeddings=True):
        super().__init__()
        weights = torch.as_tensor(word_vectors, dtype=torch.float32)
        self.embedding = torch.nn.Embedding.from_pretrained(
            weights, freeze=freeze_embeddings, padding_idx=pad_token_idx)
        self.pad_token_idx = pad_token_idx
        embedding_size = weights.shape[1]
        self.text_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embedding_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.text_logits = torch.nn.Linear(hidden_size, num_classes)
        prior = torch.as_tensor(seasonal_log_prior, dtype=torch.float32)
        if prior.shape != (4, num_classes):
            raise ValueError("seasonal_log_prior must have shape (4, num_classes)")
        self.register_buffer("seasonal_log_prior", prior)
        self.residual_scale = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, token_ids, calendar_quarters=None):
        if calendar_quarters is None:
            raise ValueError("calendar_quarters is required by the seasonal residual model")
        mask = token_ids.ne(self.pad_token_idx)
        pooled = _masked_mean(self.embedding(token_ids), mask)
        text_residual = self.text_logits(self.text_encoder(pooled))
        scale = torch.nn.functional.softplus(self.residual_scale)
        return self.seasonal_log_prior[calendar_quarters] + scale * text_residual


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
