"""LSTM fusion model for one independent observation per company-quarter."""

import torch


ARCHITECTURE_VIEWS = {
    "calendar": dict(text=False, financial=False, calendar=True),
    "text": dict(text=True, financial=False, calendar=False),
    "financial": dict(text=False, financial=True, calendar=True),
    "fusion": dict(text=True, financial=True, calendar=True),
}


class AttentionPool(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.score = torch.nn.Sequential(
            torch.nn.Linear(size, max(8, size // 2)),
            torch.nn.Tanh(),
            torch.nn.Linear(max(8, size // 2), 1, bias=False),
        )

    def forward(self, sequence):
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1)


class QuarterSequenceClassifier(torch.nn.Module):
    """Separate current-quarter text and t-1..t-4 financial LSTMs before late fusion."""

    def __init__(self, sentence_embedding_size, financial_feature_size, num_companies,
                 num_classes=4, architecture="fusion", hidden_size=64, dropout=0.25):
        super().__init__()
        if architecture not in ARCHITECTURE_VIEWS:
            raise ValueError("Unknown architecture %s" % architecture)
        self.architecture = architecture
        self.views = ARCHITECTURE_VIEWS[architecture]
        fused_sizes = []

        if self.views["text"]:
            self.text_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(sentence_embedding_size),
                torch.nn.Linear(sentence_embedding_size, hidden_size),
                torch.nn.GELU(),
            )
            self.text_lstm = torch.nn.LSTM(
                hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
            self.text_attention = AttentionPool(hidden_size)
            fused_sizes.append(hidden_size)

        if self.views["financial"]:
            self.financial_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(financial_feature_size),
                torch.nn.Linear(financial_feature_size, hidden_size // 2),
                torch.nn.GELU(),
            )
            self.financial_lstm = torch.nn.LSTM(
                hidden_size // 2, hidden_size // 2, batch_first=True)
            self.financial_attention = AttentionPool(hidden_size // 2)
            fused_sizes.append(hidden_size // 2)
            # t-4 is a strong but strictly historical seasonal feature. Exposing it directly
            # avoids forcing a tiny dataset to preserve it through four recurrent steps.
            fused_sizes.append(4)

        company_size = max(4, min(16, num_companies * 2))
        self.company_embedding = torch.nn.Embedding(num_companies, company_size)
        fused_sizes.append(company_size)
        if self.views["calendar"]:
            fused_sizes.append(4)

        fused_size = sum(fused_sizes)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(fused_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fused_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, text_sequence, financial_sequence, company_indexes, calendar_quarters):
        fused = []
        if self.views["text"]:
            text = self.text_projection(text_sequence)
            text, _ = self.text_lstm(text)
            fused.append(self.text_attention(text))
        if self.views["financial"]:
            financial = self.financial_projection(financial_sequence)
            financial, _ = self.financial_lstm(financial)
            fused.append(self.financial_attention(financial))
            fused.append(financial_sequence[:, 0, -4:])
        fused.append(self.company_embedding(company_indexes))
        if self.views["calendar"]:
            fused.append(torch.nn.functional.one_hot(
                calendar_quarters, num_classes=4).to(text_sequence.dtype))
        return self.classifier(torch.cat(fused, dim=1))
