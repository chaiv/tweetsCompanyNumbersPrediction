import unittest

import numpy as np
import torch

from classifier.MultiViewQuarterDataset import SAFE_METADATA_FEATURE_NAMES
from classifier.MultiViewQuarterModel import MultiViewQuarterClassifier


class MultiViewQuarterModelTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(17)
        self.vectors = np.random.RandomState(17).normal(size=(20, 12)).astype(np.float32)
        self.words = torch.tensor([
            [[[1, 2, 3], [4, 5, 0]]],
            [[[6, 7, 0], [8, 0, 0]]],
        ], dtype=torch.long).reshape(2, 2, 3)
        self.words_with_padding = torch.nn.functional.pad(self.words, (0, 3), value=0)
        self.sentences = torch.randn(2, 2, 8)
        self.metadata = torch.randn(2, len(SAFE_METADATA_FEATURE_NAMES))
        self.calendar = torch.tensor([0, 1], dtype=torch.long)

    def create_model(self, **overrides):
        arguments = dict(
            num_classes=4,
            metadata_size=len(SAFE_METADATA_FEATURE_NAMES),
            metadata_mean=np.zeros(len(SAFE_METADATA_FEATURE_NAMES), dtype=np.float32),
            metadata_std=np.ones(len(SAFE_METADATA_FEATURE_NAMES), dtype=np.float32),
            word_vectors=self.vectors,
            pad_token_idx=0,
            sentence_embedding_size=8,
            hidden_size=16,
            max_tweets=2,
            dropout=0.0,
        )
        arguments.update(overrides)
        return MultiViewQuarterClassifier(**arguments)

    def test_fusion_output_shape(self):
        model = self.create_model()
        output = model(self.words, self.sentences, self.metadata, self.calendar)
        self.assertEqual((2, 4), tuple(output.shape))

    def test_top2vec_view_is_padding_invariant(self):
        model = self.create_model(use_sentence=False, use_metadata=False)
        model.eval()
        with torch.no_grad():
            original = model(self.words, self.sentences, self.metadata, self.calendar)
            padded = model(self.words_with_padding, self.sentences, self.metadata, self.calendar)
        self.assertTrue(torch.allclose(original, padded, atol=1e-6))

    def test_metadata_only_model_runs_without_text_views(self):
        model = self.create_model(use_top2vec=False, use_sentence=False, use_metadata=True)
        output = model(self.words, self.sentences, self.metadata, self.calendar)
        self.assertEqual((2, 4), tuple(output.shape))

    def test_seasonal_prior_shape_is_validated(self):
        with self.assertRaises(ValueError):
            self.create_model(seasonal_log_prior=np.zeros((3, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
