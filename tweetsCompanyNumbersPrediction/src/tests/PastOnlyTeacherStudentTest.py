import unittest
from types import SimpleNamespace

import numpy as np
import torch

from classifier.LSTMNN import LSTMNN, MEAN_POOLING
from classifier.PastOnlyTeacherFeatures import (
    SUMMARY_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    compute_class_prototypes,
    summarize_teacher_outputs,
    temporal_teacher_features,
)


class PastOnlyTeacherFeaturesTest(unittest.TestCase):

    def setUp(self):
        random = np.random.RandomState(19)
        self.hidden = random.normal(size=(24, 8)).astype(np.float32)
        self.logits = random.normal(size=(24, 4)).astype(np.float32)
        self.labels = np.repeat(np.arange(4), 6)

    def test_summary_and_temporal_feature_contract(self):
        prototypes, available = compute_class_prototypes(self.hidden, self.labels)
        summary = summarize_teacher_outputs(
            self.hidden, self.logits, prototypes, available)
        self.assertEqual((len(SUMMARY_FEATURE_NAMES),), summary.shape)
        self.assertTrue(np.isfinite(summary).all())
        temporal = temporal_teacher_features({"2019Q1": summary}, "2019Q1")
        self.assertEqual((len(TEMPORAL_FEATURE_NAMES),), temporal.shape)
        np.testing.assert_allclose(temporal[len(summary):3 * len(summary)], 0.0)
        np.testing.assert_allclose(temporal[-2:], 0.0)

    def test_temporal_deltas_use_only_earlier_quarters(self):
        size = len(SUMMARY_FEATURE_NAMES)
        summaries = {
            "2018Q1": np.ones(size, dtype=np.float32),
            "2018Q4": np.full(size, 3.0, dtype=np.float32),
            "2019Q1": np.full(size, 6.0, dtype=np.float32),
        }
        temporal = temporal_teacher_features(summaries, "2019Q1")
        np.testing.assert_allclose(temporal[:size], 6.0)
        np.testing.assert_allclose(temporal[size:2 * size], 3.0)
        np.testing.assert_allclose(temporal[2 * size:3 * size], 5.0)
        np.testing.assert_allclose(temporal[-2:], 1.0)

    def test_summary_is_invariant_to_a_shared_hidden_rotation(self):
        prototypes, available = compute_class_prototypes(self.hidden, self.labels)
        original = summarize_teacher_outputs(
            self.hidden, self.logits, prototypes, available)
        orthogonal, _ = np.linalg.qr(np.random.RandomState(20).normal(size=(8, 8)))
        rotated = summarize_teacher_outputs(
            self.hidden @ orthogonal,
            self.logits,
            prototypes @ orthogonal,
            available,
        )
        np.testing.assert_allclose(original, rotated, atol=1e-5)

    def test_missing_training_class_is_explicitly_flagged(self):
        prototypes, available = compute_class_prototypes(
            self.hidden[self.labels != 3], self.labels[self.labels != 3])
        self.assertEqual(0.0, available[3])
        summary = summarize_teacher_outputs(
            self.hidden, self.logits, prototypes, available)
        np.testing.assert_allclose(summary[-4:], available)


class OriginalLSTMRepresentationTest(unittest.TestCase):

    def test_frozen_embedding_and_exposed_pre_head_representation(self):
        torch.manual_seed(23)
        vectors = SimpleNamespace(
            vectors=np.random.RandomState(23).normal(size=(12, 6)).astype(np.float32))
        model = LSTMNN(
            emb_size=6,
            word_vectors=vectors,
            num_classes=4,
            pooling=MEAN_POOLING,
            padTokenIdx=0,
            freeze_embeddings=True,
        )
        inputs = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        representation = model.encode(inputs)
        self.assertEqual((2, 256), tuple(representation.shape))
        torch.testing.assert_close(model(inputs), model.fc3(representation))
        self.assertFalse(model.embedding.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
