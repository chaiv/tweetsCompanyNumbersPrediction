import unittest

import numpy as np
import pandas as pd
import torch

from classifier.QuarterAlignedDataset import build_quarter_groups, reporting_quarters
from classifier.QuarterTextModels import (
    MeanEmbeddingClassifier,
    PackedAttentionLSTMClassifier,
    SeasonalResidualEmbeddingClassifier,
)


class QuarterAlignedDatasetTest(unittest.TestCase):

    def test_epoch_seconds_are_converted_to_local_reporting_quarter(self):
        # 31 December 2019 23:30 UTC is already 1 January 2020 in Europe/Berlin.
        epoch_seconds = pd.Series([1577835000])
        self.assertEqual(["2020Q1"], reporting_quarters(epoch_seconds).tolist())

    def test_groups_never_cross_quarter_boundaries(self):
        dataframe = pd.DataFrame({
            "post_date": [1577831400, 1577835000, 1577838600, 1577842200],
            "class": [0, 1, 1, 1],
            "body": ["a", "b", "c", "d"],
        })
        frame, groups = build_quarter_groups(dataframe, group_size=2, drop_remainder=False)
        for group in groups:
            quarters = set(frame.loc[list(group.row_indexes), "reporting_quarter"].tolist())
            self.assertEqual({group.quarter}, quarters)


class QuarterTextModelsTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.vectors = np.random.RandomState(7).normal(size=(12, 6)).astype(np.float32)
        self.pad = 0
        self.tokens = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        self.tokens_with_padding = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=torch.long)

    def assert_padding_invariant(self, model, calendar_quarters=None):
        model.eval()
        with torch.no_grad():
            original = model(self.tokens, calendar_quarters)
            padded = model(self.tokens_with_padding, calendar_quarters)
        self.assertTrue(torch.allclose(original, padded, atol=1e-6))

    def test_mean_model_is_padding_invariant(self):
        self.assert_padding_invariant(
            MeanEmbeddingClassifier(self.vectors, self.pad, num_classes=4, hidden_size=5))

    def test_packed_attention_lstm_is_padding_invariant(self):
        self.assert_padding_invariant(
            PackedAttentionLSTMClassifier(
                self.vectors, self.pad, num_classes=4, hidden_size=4, dropout=0.0))

    def test_seasonal_residual_model_is_padding_invariant(self):
        self.assert_padding_invariant(
            SeasonalResidualEmbeddingClassifier(
                self.vectors, self.pad, num_classes=4,
                seasonal_log_prior=np.zeros((4, 4), dtype=np.float32),
                hidden_size=5, dropout=0.0),
            torch.tensor([0, 1], dtype=torch.long),
        )


if __name__ == "__main__":
    unittest.main()
