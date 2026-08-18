import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from classifier.PastOnlyMultitaskTeacher import PastOnlyMultitaskTeacher
from classifier.QuarterMetadataFeatures import (
    QUARTER_METADATA_FEATURE_NAMES,
    TEMPORAL_METADATA_FEATURE_NAMES,
    build_quarter_metadata,
    temporal_metadata_features,
)
from trainEnhancedPastOnlyTeacherStudentModel import rotate_feature_block
from trainPastOnlyTeacherStudentModel import QuarterTarget


class PastOnlyMultitaskTeacherTest(unittest.TestCase):

    def test_adapter_starts_as_identity_and_embedding_remains_frozen(self):
        torch.manual_seed(31)
        vectors = SimpleNamespace(
            vectors=np.random.RandomState(31).normal(size=(14, 6)).astype(np.float32))
        model = PastOnlyMultitaskTeacher(
            emb_size=6,
            word_vectors=vectors,
            num_financial_classes=4,
            num_training_quarters=5,
            pad_token_index=0,
            adapter_size=3,
        )
        inputs = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        torch.testing.assert_close(model.embed(inputs), model.embedding(inputs))
        outputs = model.all_outputs(inputs)
        self.assertEqual((2, 4), tuple(outputs["financial"].shape))
        self.assertEqual((2, 5), tuple(outputs["quarter"].shape))
        self.assertEqual((2, 256), tuple(outputs["representation"].shape))
        self.assertFalse(model.embedding.weight.requires_grad)
        self.assertTrue(model.adapter_down.weight.requires_grad)


class QuarterMetadataFeaturesTest(unittest.TestCase):

    def test_metadata_uses_author_volume_and_time_without_targets(self):
        frame = pd.DataFrame({
            "post_date": [
                1546300800, 1548979200, 1551398400,
                1577836800, 1580515200, 1583020800,
            ],
            "body": [
                "$AAPL 10 http://a", "plain", "@writer #tag",
                "$AAPL", "20", "https://b",
            ],
            "writer": ["a", "a", "b", "a", "c", "d"],
        })
        metadata = build_quarter_metadata(frame)
        self.assertEqual({"2019Q1", "2020Q1"}, set(metadata))
        self.assertEqual(
            (len(QUARTER_METADATA_FEATURE_NAMES),), metadata["2019Q1"].shape)
        self.assertTrue(np.isfinite(metadata["2019Q1"]).all())
        self.assertGreater(metadata["2019Q1"][4], 0.5)

    def test_temporal_metadata_uses_only_previous_and_year_ago(self):
        size = len(QUARTER_METADATA_FEATURE_NAMES)
        metadata = {
            "2018Q1": np.ones(size, dtype=np.float32),
            "2018Q4": np.full(size, 3.0, dtype=np.float32),
            "2019Q1": np.full(size, 6.0, dtype=np.float32),
        }
        result = temporal_metadata_features(metadata, "2019Q1")
        self.assertEqual((len(TEMPORAL_METADATA_FEATURE_NAMES),), result.shape)
        np.testing.assert_allclose(result[:size], 6.0)
        np.testing.assert_allclose(result[size:2 * size], 3.0)
        np.testing.assert_allclose(result[2 * size:3 * size], 5.0)
        np.testing.assert_allclose(result[-2:], 1.0)


class EnhancedShuffleControlTest(unittest.TestCase):

    def test_shuffle_rotates_only_requested_feature_block_within_company(self):
        matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
        targets = [
            QuarterTarget("a", 0, "2019Q1", 0, -1.0, 0.0),
            QuarterTarget("a", 0, "2019Q2", 1, 1.0, 0.0),
            QuarterTarget("b", 1, "2019Q1", 2, 20.0, 0.0),
            QuarterTarget("b", 1, "2019Q2", 3, 40.0, 0.0),
        ]
        shuffled = rotate_feature_block(matrix, targets, 1, 4, seed=9)
        np.testing.assert_allclose(shuffled[:, 0], matrix[:, 0])
        np.testing.assert_allclose(shuffled[:, 4:], matrix[:, 4:])
        np.testing.assert_allclose(shuffled[[0, 1], 1:4], matrix[[1, 0], 1:4])
        np.testing.assert_allclose(shuffled[[2, 3], 1:4], matrix[[3, 2], 1:4])


if __name__ == "__main__":
    unittest.main()
