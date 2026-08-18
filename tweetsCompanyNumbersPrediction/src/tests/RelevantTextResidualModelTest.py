import unittest

import numpy as np
import pandas as pd
import torch

from classifier.ExplainableResidualQuarterModel import ExplainableResidualQuarterModel
from classifier.RelevantQuarterTextDataset import RelevantQuarterRecord, shuffle_record_text
from featureinterpretation.HierarchicalQuarterAttributions import (
    HierarchicalQuarterAttributions,
    aggregate_topic_attributions,
)


class RelevantTextResidualModelTest(unittest.TestCase):

    def setUp(self):
        self.vectors = np.random.RandomState(3).normal(size=(20, 12)).astype(np.float32)
        self.model = ExplainableResidualQuarterModel(
            self.vectors, pad_token_idx=0, financial_feature_size=12,
            hidden_size=16, dropout=0.0, modality_dropout=0.0)
        self.model.eval()
        self.words = torch.tensor([
            [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 0]],
            [[2, 3, 0, 0, 0], [5, 6, 7, 0, 0], [8, 9, 0, 0, 0]],
        ])
        self.financial = torch.zeros(2, 4, 12)
        self.financial[0, 0, -4] = 1
        self.financial[1, 0, -1] = 1

    def test_heads_return_four_quarter_classes(self):
        logits = self.model.all_logits(
            self.words, self.financial, apply_modality_dropout=False)
        self.assertEqual({"finance", "text", "fusion"}, set(logits))
        for values in logits.values():
            self.assertEqual((2, 4), tuple(values.shape))

    def test_padding_does_not_change_text_logits(self):
        padded = torch.nn.functional.pad(self.words, (0, 3), value=0)
        with torch.no_grad():
            original = self.model.text_logits(self.words)
            changed = self.model.text_logits(padded)
        torch.testing.assert_close(original, changed, atol=1e-6, rtol=1e-6)

    def test_financial_base_prefers_strictly_lagged_class(self):
        logits = self.model.all_logits(
            self.words, self.financial, apply_modality_dropout=False)["finance"]
        self.assertEqual([0, 3], logits.argmax(dim=1).tolist())

    def test_integrated_gradients_returns_raw_token_and_tweet_contributions(self):
        result = HierarchicalQuarterAttributions(self.model).attribute(
            self.words[:1], target=1, n_steps=8)
        self.assertEqual((1, 3, 5), tuple(result["token_signed"].shape))
        self.assertEqual((1, 3), tuple(result["tweet_signed"].shape))
        self.assertTrue(torch.isfinite(result["token_signed"]).all())
        self.assertTrue(torch.equal(
            result["token_signed"].masked_select(self.words[:1].eq(0)),
            torch.zeros(6),
        ))


class RelevantTextUtilitiesTest(unittest.TestCase):

    def test_shuffle_breaks_text_alignment_but_preserves_quarter_targets(self):
        records = [RelevantQuarterRecord(
            quarter="2019Q%d" % (index + 1), label=index,
            word_ids=np.full((1, 2, 3), index + 1),
            tweet_ids=np.full((1, 2), index + 10),
            financial_sequence=np.zeros((4, 12), dtype=np.float32),
        ) for index in range(4)]
        shuffled = shuffle_record_text(records, seed=7)
        self.assertEqual([record.quarter for record in records],
                         [record.quarter for record in shuffled])
        self.assertEqual([record.label for record in records],
                         [record.label for record in shuffled])
        self.assertTrue(all(not np.array_equal(left.word_ids, right.word_ids)
                            for left, right in zip(records, shuffled)))

    def test_topic_aggregation_keeps_signed_and_absolute_contributions(self):
        attributions = pd.DataFrame({
            "quarter": ["2019Q1", "2019Q1"],
            "target_class": [1, 1],
            "tweet_id": [10, 11],
            "token": ["sales", "risk"],
            "token_attribution": [0.4, -0.2],
            "token_attribution_abs": [0.4, 0.2],
        })
        topics = pd.DataFrame({"tweet_id": [10, 11], "topic_id": [3, 3]})
        result = aggregate_topic_attributions(attributions, topics)
        self.assertAlmostEqual(0.2, result.iloc[0]["token_attribution"])
        self.assertAlmostEqual(0.6, result.iloc[0]["token_attribution_abs"])
        self.assertEqual(2, result.iloc[0]["tweet_count"])


if __name__ == "__main__":
    unittest.main()
