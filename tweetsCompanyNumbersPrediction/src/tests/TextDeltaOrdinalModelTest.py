import unittest

import numpy as np
import torch

from classifier.TextDeltaOrdinalModel import (
    TextDeltaOrdinalModel,
    normalized_logits,
    ordinal_probabilities,
    ordinal_targets,
)
from classifier.TextDeltaQuarterDataset import add_quarter_deltas


class TextDeltaQuarterDatasetTest(unittest.TestCase):

    def test_quarter_deltas_use_only_previous_and_year_ago_text(self):
        features = {
            "2018Q1": np.ones((2, 3), dtype=np.float32),
            "2018Q4": np.full((2, 3), 3.0, dtype=np.float32),
            "2019Q1": np.full((2, 3), 6.0, dtype=np.float32),
        }
        result = add_quarter_deltas(features, "2019Q1")
        self.assertEqual((2, 11), result.shape)
        np.testing.assert_allclose(result[:, :3], 6.0)
        np.testing.assert_allclose(result[:, 3:6], 3.0)
        np.testing.assert_allclose(result[:, 6:9], 5.0)
        np.testing.assert_allclose(result[:, 9:], 1.0)

    def test_missing_history_is_zero_and_flagged(self):
        result = add_quarter_deltas(
            {"2015Q1": np.ones((2, 3), dtype=np.float32)}, "2015Q1")
        np.testing.assert_allclose(result[:, 3:9], 0.0)
        np.testing.assert_allclose(result[:, 9:], 0.0)


class TextDeltaOrdinalModelTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(9)
        self.model = TextDeltaOrdinalModel(
            text_feature_size=20, financial_feature_size=12,
            num_companies=3, hidden_size=24, dropout=0.0, text_weight=0.4)
        self.model.eval()
        self.text = torch.randn(5, 8, 20)
        self.financial = torch.randn(5, 4, 12)
        self.company = torch.tensor([0, 1, 2, 0, 1])
        self.quarter = torch.tensor([0, 1, 2, 3, 0])

    def test_ordinal_targets_encode_ordered_thresholds(self):
        targets = ordinal_targets(torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(targets, torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]))

    def test_ordinal_probabilities_are_valid(self):
        threshold_logits = torch.tensor([[2.0, 0.0, -2.0]])
        probabilities = ordinal_probabilities(threshold_logits)
        self.assertTrue(torch.all(probabilities > 0))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))

    def test_all_heads_and_auxiliary_outputs_have_expected_shapes(self):
        outputs = self.model.all_outputs(
            self.text, self.financial, self.company, self.quarter)
        self.assertEqual((5, 4), tuple(outputs["text"].shape))
        self.assertEqual((5, 4), tuple(outputs["finance"].shape))
        self.assertEqual((5, 4), tuple(outputs["fusion"].shape))
        self.assertEqual((5, 3), tuple(outputs["ordinal"].shape))
        self.assertEqual((5,), tuple(outputs["regression"].shape))

    def test_fusion_keeps_exact_requested_text_fraction(self):
        outputs = self.model.all_outputs(
            self.text, self.financial, self.company, self.quarter)
        expected = (
            0.4 * normalized_logits(outputs["text"])
            + 0.6 * normalized_logits(outputs["finance"])
        )
        torch.testing.assert_close(outputs["fusion"], expected)


if __name__ == "__main__":
    unittest.main()
