"""Hierarchical multi-view classifier for groups of financial tweets."""

import torch


def _masked_word_mean(embeddings, mask):
    weights = mask.unsqueeze(-1).to(embeddings.dtype)
    return (embeddings * weights).sum(dim=2) / weights.sum(dim=2).clamp(min=1)


class MultiViewQuarterClassifier(torch.nn.Module):
    """Fuse Top2Vec words, MiniLM tweet semantics and safe metadata with tweet attention."""

    def __init__(self, num_classes, metadata_size, metadata_mean, metadata_std,
                 use_top2vec=True, use_sentence=True, use_metadata=True,
                 word_vectors=None, pad_token_idx=None, sentence_embedding_size=384,
                 hidden_size=128, max_tweets=20, dropout=0.3, seasonal_log_prior=None):
        super().__init__()
        if not (use_top2vec or use_sentence or use_metadata):
            raise ValueError("At least one model view must be enabled")
        self.use_top2vec = use_top2vec
        self.use_sentence = use_sentence
        self.use_metadata = use_metadata
        self.pad_token_idx = pad_token_idx

        tweet_view_sizes = []
        if use_top2vec:
            if word_vectors is None or pad_token_idx is None:
                raise ValueError("Top2Vec view requires word_vectors and pad_token_idx")
            weights = torch.as_tensor(word_vectors, dtype=torch.float32)
            self.word_embedding = torch.nn.Embedding.from_pretrained(
                weights, freeze=True, padding_idx=pad_token_idx)
            self.word_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(weights.shape[1]),
                torch.nn.Linear(weights.shape[1], hidden_size),
                torch.nn.GELU(),
            )
            tweet_view_sizes.append(hidden_size)
        if use_sentence:
            self.sentence_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(sentence_embedding_size),
                torch.nn.Linear(sentence_embedding_size, hidden_size),
                torch.nn.GELU(),
            )
            tweet_view_sizes.append(hidden_size)

        self.has_text_view = bool(tweet_view_sizes)
        if self.has_text_view:
            self.tweet_fusion = torch.nn.Sequential(
                torch.nn.Linear(sum(tweet_view_sizes), hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            self.position_embedding = torch.nn.Parameter(torch.zeros(1, max_tweets, hidden_size))
            torch.nn.init.normal_(self.position_embedding, std=0.02)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.tweet_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.tweet_attention = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size // 2, 1, bias=False),
            )
            self.text_output = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),
                torch.nn.GELU(),
            )

        if use_metadata:
            mean = torch.as_tensor(metadata_mean, dtype=torch.float32)
            std = torch.as_tensor(metadata_std, dtype=torch.float32).clamp(min=1e-6)
            if mean.numel() != metadata_size or std.numel() != metadata_size:
                raise ValueError("Metadata statistics have the wrong size")
            self.register_buffer("metadata_mean", mean)
            self.register_buffer("metadata_std", std)
            self.metadata_encoder = torch.nn.Sequential(
                torch.nn.Linear(metadata_size, 64),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(64, 64),
                torch.nn.GELU(),
            )

        fused_size = (hidden_size if self.has_text_view else 0) + (64 if use_metadata else 0)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(fused_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fused_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

        if seasonal_log_prior is None:
            self.seasonal_log_prior = None
        else:
            prior = torch.as_tensor(seasonal_log_prior, dtype=torch.float32)
            if prior.shape != (4, num_classes):
                raise ValueError("seasonal_log_prior must have shape (4, num_classes)")
            self.register_buffer("seasonal_log_prior", prior)
            self.residual_scale = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, word_ids, sentence_embeddings, metadata, calendar_quarters):
        views = []
        if self.use_top2vec:
            word_mask = word_ids.ne(self.pad_token_idx)
            embedded = self.word_embedding(word_ids)
            views.append(self.word_projection(_masked_word_mean(embedded, word_mask)))
        if self.use_sentence:
            views.append(self.sentence_projection(sentence_embeddings))

        fused = []
        if self.has_text_view:
            tweet_vectors = self.tweet_fusion(torch.cat(views, dim=-1))
            positions = self.position_embedding[:, :tweet_vectors.shape[1]]
            encoded_tweets = self.tweet_encoder(tweet_vectors + positions)
            attention = torch.softmax(self.tweet_attention(encoded_tweets).squeeze(-1), dim=1)
            attended = (encoded_tweets * attention.unsqueeze(-1)).sum(dim=1)
            mean = encoded_tweets.mean(dim=1)
            fused.append(self.text_output(torch.cat([attended, mean], dim=1)))
        if self.use_metadata:
            normalized = (metadata - self.metadata_mean) / self.metadata_std
            fused.append(self.metadata_encoder(normalized))

        logits = self.classifier(torch.cat(fused, dim=1))
        if self.seasonal_log_prior is not None:
            scale = torch.nn.functional.softplus(self.residual_scale)
            logits = self.seasonal_log_prior[calendar_quarters] + scale * logits
        return logits
