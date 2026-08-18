"""Token-preserving hierarchical text model with a strictly lagged financial base."""

import torch


def masked_attention(sequence, mask, scorer):
    scores = scorer(sequence).squeeze(-1).masked_fill(~mask, -1e4)
    weights = torch.softmax(scores, dim=1)
    return (sequence * weights.unsqueeze(-1)).sum(dim=1), weights


class ExplainableResidualQuarterModel(torch.nn.Module):
    """Top2Vec token BiLSTM -> tweet BiLSTM -> residual over t-4 seasonal logits."""

    def __init__(self, word_vectors, pad_token_idx, financial_feature_size, num_classes=4,
                 hidden_size=96, dropout=0.3, modality_dropout=0.25):
        super().__init__()
        weights = torch.as_tensor(word_vectors, dtype=torch.float32)
        self.word_embedding = torch.nn.Embedding.from_pretrained(
            weights, freeze=True, padding_idx=pad_token_idx)
        self.pad_token_idx = pad_token_idx
        self.modality_dropout = modality_dropout
        self.num_classes = num_classes

        self.word_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(weights.shape[1]),
            torch.nn.Linear(weights.shape[1], hidden_size),
            torch.nn.GELU(),
        )
        self.token_lstm = torch.nn.LSTM(
            hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.token_attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.Tanh(),
            torch.nn.Linear(hidden_size // 2, 1, bias=False))
        self.tweet_lstm = torch.nn.LSTM(
            hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.tweet_attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.Tanh(),
            torch.nn.Linear(hidden_size // 2, 1, bias=False))
        self.text_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_size, num_classes))

        self.financial_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(financial_feature_size),
            torch.nn.Linear(financial_feature_size, hidden_size // 2), torch.nn.GELU())
        self.financial_lstm = torch.nn.LSTM(
            hidden_size // 2, hidden_size // 2, batch_first=True)
        self.financial_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size // 2), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, num_classes))
        self.finance_residual_scale = torch.nn.Parameter(torch.tensor(-2.0))
        self.text_residual_scale = torch.nn.Parameter(torch.tensor(-2.0))

    def encode_text(self, word_ids, return_attention=False):
        batch_size, tweets, words = word_ids.shape
        token_mask = word_ids.ne(self.pad_token_idx)
        flattened_mask = token_mask.reshape(batch_size * tweets, words)
        # Keep the quarter batch dimension at the attributed embedding layer. Captum's
        # LayerIntegratedGradients then returns [quarter, tweet, token, embedding]
        # instead of incorrectly treating every tweet as an independent example.
        embedded = self.word_projection(self.word_embedding(word_ids))
        embedded = embedded.reshape(batch_size * tweets, words, -1)
        lengths = flattened_mask.sum(dim=1).clamp(min=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_tokens, _ = self.token_lstm(packed)
        encoded_tokens, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_tokens, batch_first=True, total_length=words)
        tweet_vectors, token_attention = masked_attention(
            encoded_tokens, flattened_mask, self.token_attention)
        tweet_vectors = tweet_vectors.reshape(batch_size, tweets, -1)
        encoded_tweets, _ = self.tweet_lstm(tweet_vectors)
        tweet_mask = token_mask.any(dim=2)
        quarter_vector, tweet_attention = masked_attention(
            encoded_tweets, tweet_mask, self.tweet_attention)
        if return_attention:
            return quarter_vector, token_attention.reshape(batch_size, tweets, words), tweet_attention
        return quarter_vector

    def text_logits(self, word_ids):
        return self.text_head(self.encode_text(word_ids))

    def all_logits(self, word_ids, financial_sequence, apply_modality_dropout=True):
        text_logits = self.text_logits(word_ids)
        financial = self.financial_projection(financial_sequence)
        _, (hidden, _) = self.financial_lstm(financial)
        lag_class = financial_sequence[:, 0, -4:]
        seasonal_logits = 2.2 * lag_class
        finance_scale = torch.nn.functional.softplus(self.finance_residual_scale)
        text_scale = torch.nn.functional.softplus(self.text_residual_scale)
        finance_logits = seasonal_logits + finance_scale * self.financial_head(hidden[-1])
        fused_logits = finance_logits + text_scale * text_logits
        if self.training and apply_modality_dropout and self.modality_dropout > 0:
            drop_finance = torch.rand(
                fused_logits.shape[0], 1, device=fused_logits.device) < self.modality_dropout
            fused_logits = torch.where(drop_finance, text_logits, fused_logits)
        return {
            "fusion": fused_logits,
            "text": text_logits,
            "finance": finance_logits,
        }

    def forward(self, word_ids, financial_sequence):
        return self.all_logits(word_ids, financial_sequence)["fusion"]
