"""Text-delta multi-task model with ordered classes and fixed-strength late fusion."""

import math

import torch

from classifier.QuarterSequenceModel import AttentionPool


def ordinal_targets(labels, num_classes=4):
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).to(torch.float32)


def ordinal_probabilities(threshold_logits, epsilon=1e-6):
    """Convert monotonically decreasing P(y > k) values to class probabilities."""
    cumulative = torch.sigmoid(threshold_logits)
    probabilities = torch.cat((
        1.0 - cumulative[:, :1],
        cumulative[:, :-1] - cumulative[:, 1:],
        cumulative[:, -1:],
    ), dim=1)
    probabilities = probabilities.clamp_min(epsilon)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def normalized_logits(logits, epsilon=1e-5):
    return (logits - logits.mean(dim=1, keepdim=True)) / (
        logits.std(dim=1, keepdim=True, unbiased=False) + epsilon)


class OrderedLogitHead(torch.nn.Module):

    def __init__(self, input_size, num_classes=4):
        super().__init__()
        self.risk = torch.nn.Linear(input_size, 1)
        self.first_cutpoint = torch.nn.Parameter(torch.tensor(-1.0))
        inverse_softplus_one = math.log(math.exp(1.0) - 1.0)
        self.cutpoint_gaps = torch.nn.Parameter(
            torch.full((num_classes - 2,), inverse_softplus_one))

    def cutpoints(self):
        gaps = torch.nn.functional.softplus(self.cutpoint_gaps)
        return torch.cat((
            self.first_cutpoint.reshape(1),
            self.first_cutpoint + torch.cumsum(gaps, dim=0),
        ))

    def forward(self, features):
        return self.risk(features) - self.cutpoints().unsqueeze(0)


class TextDeltaOrdinalModel(torch.nn.Module):
    """GRU text-change encoder plus a strictly lagged financial encoder.

    Text and financial logits are normalized before late fusion, so the requested text weight is
    an actual contribution fraction rather than a scale that can silently collapse during fitting.
    """

    def __init__(self, text_feature_size, financial_feature_size, num_companies=3,
                 num_classes=4, hidden_size=96, dropout=0.3, text_weight=0.4):
        super().__init__()
        if not 0.0 <= text_weight <= 1.0:
            raise ValueError("text_weight must be between zero and one")
        self.text_weight = float(text_weight)
        self.num_classes = num_classes
        company_size = max(6, min(16, num_companies * 3))
        self.company_embedding = torch.nn.Embedding(num_companies, company_size)

        self.text_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(text_feature_size),
            torch.nn.Linear(text_feature_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.text_gru = torch.nn.GRU(
            hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.text_attention = AttentionPool(hidden_size)
        context_size = hidden_size + company_size + 4
        self.text_trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(context_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(context_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.ordinal_head = OrderedLogitHead(hidden_size, num_classes=num_classes)
        self.regression_head = torch.nn.Linear(hidden_size, 1)

        self.financial_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(financial_feature_size),
            torch.nn.Linear(financial_feature_size, hidden_size // 2),
            torch.nn.GELU(),
        )
        self.financial_gru = torch.nn.GRU(
            hidden_size // 2, hidden_size // 2, batch_first=True)
        finance_size = hidden_size // 2 + 4 + company_size + 4
        self.financial_head = torch.nn.Sequential(
            torch.nn.LayerNorm(finance_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(finance_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 2, num_classes),
        )

    def text_outputs(self, text_sequence, company_indexes, calendar_quarters):
        text = self.text_projection(text_sequence)
        text, _ = self.text_gru(text)
        text = self.text_attention(text)
        calendar = torch.nn.functional.one_hot(
            calendar_quarters, num_classes=4).to(text.dtype)
        context = self.text_trunk(torch.cat((
            text, self.company_embedding(company_indexes), calendar
        ), dim=1))
        ordinal_logits = self.ordinal_head(context)
        probabilities = ordinal_probabilities(ordinal_logits)
        return torch.log(probabilities), ordinal_logits, self.regression_head(context).squeeze(1)

    def finance_logits(self, financial_sequence, company_indexes, calendar_quarters):
        financial = self.financial_projection(financial_sequence)
        _, hidden = self.financial_gru(financial)
        lag_class = financial_sequence[:, 0, -4:]
        calendar = torch.nn.functional.one_hot(
            calendar_quarters, num_classes=4).to(financial.dtype)
        learned = self.financial_head(torch.cat((
            hidden[-1], lag_class, self.company_embedding(company_indexes), calendar
        ), dim=1))
        return 2.2 * lag_class + learned

    def all_outputs(self, text_sequence, financial_sequence, company_indexes, calendar_quarters):
        text_logits, ordinal_logits, regression = self.text_outputs(
            text_sequence, company_indexes, calendar_quarters)
        finance_logits = self.finance_logits(
            financial_sequence, company_indexes, calendar_quarters)
        fusion_logits = (
            self.text_weight * normalized_logits(text_logits)
            + (1.0 - self.text_weight) * normalized_logits(finance_logits)
        )
        return {
            "text": text_logits,
            "finance": finance_logits,
            "fusion": fusion_logits,
            "ordinal": ordinal_logits,
            "regression": regression,
        }

    def forward(self, text_sequence, financial_sequence, company_indexes, calendar_quarters):
        return self.all_outputs(
            text_sequence, financial_sequence, company_indexes, calendar_quarters)["fusion"]
