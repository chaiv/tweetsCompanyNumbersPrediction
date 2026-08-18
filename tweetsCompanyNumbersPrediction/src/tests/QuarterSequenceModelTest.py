import unittest

import numpy as np
import pandas as pd
import torch

from classifier.QuarterSequenceDataset import (
    FINANCIAL_FEATURE_NAMES,
    lagged_financial_sequence,
    percent_change_class,
    prepare_financial_quarters,
    select_text_bags,
)
from classifier.QuarterSequenceModel import QuarterSequenceClassifier
from trainQuarterSequenceModel import shuffle_text_within_company
from classifier.QuarterSequenceDataset import QuarterSequenceRecord


class QuarterSequenceDatasetTest(unittest.TestCase):

    def setUp(self):
        self.financial = prepare_financial_quarters(pd.DataFrame({
            "from_date": ["01/10/2018", "01/01/2019", "01/04/2019", "01/07/2019",
                          "01/10/2019"],
            "value": [100.0, 120.0, 90.0, 110.0, 130.0],
            "percent_change": [np.nan, 20.0, -25.0, 22.2, 18.2],
        }))

    def test_multiclass_boundaries(self):
        self.assertEqual([0, 1, 1, 2, 2, 3], [
            percent_change_class(value) for value in (-0.01, 0.0, 15.0, 15.01, 30.0, 30.01)])

    def test_financial_history_excludes_target_value_and_change(self):
        sequence = lagged_financial_sequence(self.financial, "2019Q4", lookback=4)
        changed_target = self.financial.copy()
        changed_target.loc[changed_target["quarter"] == "2019Q4", ["value", "percent_change"]] = [
            999999.0, -99.0]
        changed = lagged_financial_sequence(changed_target, "2019Q4", lookback=4)
        np.testing.assert_allclose(sequence, changed)
        self.assertEqual((4, len(FINANCIAL_FEATURE_NAMES)), sequence.shape)

    def test_lag_classes_are_derived_only_from_historical_changes(self):
        sequence = lagged_financial_sequence(self.financial, "2019Q4", lookback=4)
        # 2018Q4 has no percentage change in the fixture, hence no known lag class.
        np.testing.assert_array_equal(np.zeros(4), sequence[0, -4:])
        # The latest lag is 2019Q3 (+22.2%), which is class 2.
        np.testing.assert_array_equal([0, 0, 1, 0], sequence[-1, -4:])

    def test_text_bags_cover_each_chronological_bin(self):
        frame = pd.DataFrame({
            "post_date": np.arange(16),
            "body": [str(index) for index in range(16)],
            "reporting_quarter": ["2019Q1"] * 16,
        })
        bags = select_text_bags(
            frame, "2019Q1", bins=4, tweets_per_bin=2, variants=2, seed=7)
        self.assertEqual((2, 4, 2), np.asarray(bags).shape)
        for variant in bags:
            for bin_index, values in enumerate(variant):
                self.assertTrue(all(bin_index * 4 <= int(value) < (bin_index + 1) * 4
                                    for value in values))

    def test_shuffled_text_control_preserves_targets_but_changes_quarter_text(self):
        records = [QuarterSequenceRecord(
            company="example",
            company_index=0,
            quarter="2019Q%d" % (index + 1),
            label=index,
            text_sequences=np.full((1, 2, 3), index, dtype=np.float32),
            financial_sequence=np.zeros((4, len(FINANCIAL_FEATURE_NAMES)), dtype=np.float32),
        ) for index in range(4)]
        shuffled = shuffle_text_within_company(records, seed=3)
        self.assertEqual([record.label for record in records], [record.label for record in shuffled])
        self.assertEqual([record.quarter for record in records], [record.quarter for record in shuffled])
        self.assertTrue(all(not np.array_equal(original.text_sequences, changed.text_sequences)
                            for original, changed in zip(records, shuffled)))


class QuarterSequenceModelTest(unittest.TestCase):

    def test_all_ablation_models_return_quarter_class_logits(self):
        text = torch.randn(3, 8, 12)
        financial = torch.randn(3, 4, len(FINANCIAL_FEATURE_NAMES))
        companies = torch.tensor([0, 1, 2])
        quarters = torch.tensor([0, 1, 2])
        for architecture in ("calendar", "text", "financial", "fusion"):
            model = QuarterSequenceClassifier(
                sentence_embedding_size=12,
                financial_feature_size=len(FINANCIAL_FEATURE_NAMES),
                num_companies=3,
                architecture=architecture,
                hidden_size=16,
                dropout=0.0,
            )
            self.assertEqual((3, 4), tuple(model(text, financial, companies, quarters).shape))


if __name__ == "__main__":
    unittest.main()
