import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from classifier.PureTextQuarterViews import (
    ImportantWordLogOdds,
    QuarterStableImportantWords,
    SparseOrdinalClassifier,
    STYLE_FEATURE_NAMES,
    aggregate_temporal_quarter_probabilities,
    aggregate_quarter_probabilities,
    finance_relevant_semantic_text,
    group_style_features,
    normalize_semantic_text,
)
from trainPureTextQuarterModel import (
    VIEWS,
    TextQuarterTarget,
    fuse_probabilities,
    fusion_weight_grid,
    shuffle_quarter_probabilities,
)
from classifier.QualityFilteredQuarterGroups import quality_filter_tweets
from classifier.NumericQuarterTextFeatures import (
    NUMERIC_FEATURE_NAMES,
    numeric_quarter_features,
    percentage_signal_probabilities,
)
from trainPooledTextDeltaQuarterModel import previous_quarter
from trainNumericTextSignalQuarterModel import paired_accuracy_audit, tesla_conflict_gate


class PureTextQuarterViewsTest(unittest.TestCase):

    def test_tesla_conflict_gate_only_changes_declared_conflicts(self):
        rows = [("tesla", "2018Q1"), ("tesla", "2018Q4"), ("amazon", "2018Q1")]
        base = np.asarray([
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        numeric = np.asarray([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        result = tesla_conflict_gate(rows, base, numeric, [40.0, -30.0, 40.0])
        np.testing.assert_array_equal(result.argmax(axis=1), [1, 3, 3])

    def test_paired_accuracy_audit_counts_discordant_quarters(self):
        primary = {"true": [0, 1, 2, 3], "predicted": [0, 1, 2, 0]}
        control = {"true": [0, 1, 2, 3], "predicted": [1, 1, 0, 3]}
        audit = paired_accuracy_audit(primary, control)
        self.assertEqual(2, audit["primary_only_correct"])
        self.assertEqual(1, audit["control_only_correct"])
        self.assertEqual(3, audit["correct_company_quarters"])
        self.assertEqual(4, audit["total_company_quarters"])

    def test_numeric_text_features_preserve_target_context_and_direction(self):
        features = numeric_quarter_features([
            "Amazon reported revenue up 22% to $63.4 billion.",
            "AMZN net sales declined 4 percent to 59.7 billion.",
            "The stock price rose 80% but this sentence has no company metric.",
        ], "amazon", total_tweets=100)
        probabilities = percentage_signal_probabilities(features)
        self.assertEqual((len(NUMERIC_FEATURE_NAMES),), features.shape)
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(probabilities[0], probabilities[3])
        self.assertGreater(probabilities[2], probabilities[3])
        self.assertAlmostEqual(1.0, float(probabilities.sum()))

    def test_previous_quarter_handles_year_boundary_and_year_over_year_lag(self):
        self.assertEqual("2018Q4", previous_quarter("2019Q1"))
        self.assertEqual("2018Q1", previous_quarter("2019Q1", lag=4))

    def test_sparse_ordinal_classifier_returns_valid_monotonic_probabilities(self):
        features = csr_matrix(np.asarray([
            [-3.0, 1.0], [-2.5, 0.5], [-2.0, 0.0],
            [-1.0, 1.0], [-0.5, 0.5], [-0.2, 0.0],
            [0.2, 0.0], [0.5, 0.5], [1.0, 1.0],
            [2.0, 0.0], [2.5, 0.5], [3.0, 1.0],
        ], dtype=np.float32))
        labels = np.repeat(np.arange(4), 3)
        classifier = SparseOrdinalClassifier(
            regularization=4.0, seed=7).fit(features, labels)

        thresholds = classifier.threshold_probabilities(features)
        probabilities = classifier.predict_proba(features)
        direction = classifier.predict_direction_proba(features)

        self.assertEqual((12, 3), thresholds.shape)
        self.assertTrue((thresholds[:, :-1] >= thresholds[:, 1:]).all())
        self.assertTrue((probabilities >= 0.0).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(direction.sum(axis=1), 1.0)
        self.assertLess(probabilities[0].argmax(), probabilities[-1].argmax())

    def test_group_probabilities_are_averaged_once_per_quarter(self):
        probabilities = np.asarray([
            [0.8, 0.2, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.7],
        ])
        aggregated = aggregate_quarter_probabilities(
            ["2019Q1", "2019Q1", "2019Q2"], probabilities,
            ["2019Q1", "2019Q2"])
        np.testing.assert_allclose(aggregated[0], [0.6, 0.4, 0.0, 0.0])
        np.testing.assert_allclose(aggregated[1], [0.0, 0.0, 0.3, 0.7])

    def test_temporal_aggregation_can_use_late_quarter_evidence(self):
        probabilities = np.asarray([
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.8],
            [0.0, 0.0, 0.1, 0.9],
        ])
        late = aggregate_temporal_quarter_probabilities(
            ["2019Q1"] * 4, [1, 2, 3, 4], probabilities, ["2019Q1"],
            mode="late_half_mean")
        self.assertEqual(3, int(late.argmax(axis=1)[0]))
        vote = aggregate_temporal_quarter_probabilities(
            ["2019Q1"] * 4, [1, 2, 3, 4], probabilities, ["2019Q1"],
            mode="late_half_vote", temperature=0.5)
        self.assertEqual(3, int(vote.argmax(axis=1)[0]))
        np.testing.assert_allclose(late.sum(axis=1), 1.0)

    def test_past_only_log_odds_identifies_class_specific_words(self):
        texts = [
            "loss decline weak", "loss falling", "steady normal", "steady ordinary",
            "growth rising", "growth better", "surge record", "surge exceptional",
        ]
        labels = [0, 0, 1, 1, 2, 2, 3, 3]
        model = ImportantWordLogOdds(max_features=100).fit(texts, labels)
        top = model.top_words(count=10)
        self.assertIn("loss", [value["token"] for value in top["0"]])
        self.assertIn("surge", [value["token"] for value in top["3"]])
        probabilities = model.predict_proba(["surge record"], temperature=0.5)
        self.assertEqual(3, int(probabilities.argmax(axis=1)[0]))

    def test_style_features_are_target_independent_and_finite(self):
        features = group_style_features(
            ["$AAPL will rise 10%!", "Perhaps uncertain? http://example.com"],
            ["author-a", "author-b"],
            [100, 200],
        )
        self.assertEqual((len(STYLE_FEATURE_NAMES),), features.shape)
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[12], 0.0)
        self.assertGreater(features[14], 0.0)

    def test_semantic_normalization_removes_period_and_source_shortcuts(self):
        normalized = normalize_semantic_text(
            "$AAPL earnings grew in September 2018 https://owler.com/x?utm_source=test")
        self.assertIn("earnings", normalized)
        self.assertIn("grew", normalized)
        self.assertNotIn("2018", normalized)
        self.assertNotIn("september", normalized)
        self.assertNotIn("owler", normalized)
        self.assertNotIn("aapl", normalized)

    def test_finance_relevant_text_prefers_event_bearing_tweets(self):
        selected = finance_relevant_semantic_text([
            "beautiful new phone color", "quarterly revenue beat estimates",
        ])
        self.assertIn("revenue", selected)
        self.assertNotIn("beautiful", selected)

    def test_quarter_stable_words_require_recurrence_across_class_quarters(self):
        texts = [
            "recurring growth alpha", "recurring growth beta",
            "ordinary stable gamma", "ordinary stable delta",
            "recurring growth epsilon", "recurring growth zeta",
            "ordinary stable eta", "ordinary stable theta",
        ]
        labels = [2, 2, 1, 1, 2, 2, 1, 1]
        quarters = [
            "2015Q1", "2015Q1", "2015Q2", "2015Q2",
            "2016Q1", "2016Q1", "2016Q2", "2016Q2",
        ]
        model = QuarterStableImportantWords(max_features=100).fit(
            texts, labels, quarters)
        top = model.top_words(count=10)
        self.assertIn("growth", [value["token"] for value in top["2"]])
        self.assertNotIn("alpha", [value["token"] for value in top["2"]])
        prediction = model.predict_proba(["recurring growth"], temperature=0.5)
        self.assertEqual(2, int(prediction.argmax(axis=1)[0]))


class PureTextFusionTest(unittest.TestCase):

    def test_weight_grid_contains_only_convex_view_combinations(self):
        weights = list(fusion_weight_grid(0.25))
        self.assertTrue(weights)
        for values in weights:
            self.assertEqual(len(VIEWS), len(values))
            self.assertAlmostEqual(1.0, float(values.sum()))
            self.assertTrue((values >= 0).all())

    def test_fusion_and_shuffle_preserve_probability_contract(self):
        view_probabilities = {
            view: np.eye(4, dtype=np.float64) for view in VIEWS
        }
        fused = fuse_probabilities(
            view_probabilities, np.full(len(VIEWS), 1.0 / len(VIEWS)))
        np.testing.assert_allclose(fused.sum(axis=1), 1.0)
        targets = [
            TextQuarterTarget("a", "2019Q1", 0),
            TextQuarterTarget("a", "2019Q2", 1),
            TextQuarterTarget("b", "2019Q1", 2),
            TextQuarterTarget("b", "2019Q2", 3),
        ]
        shuffled = shuffle_quarter_probabilities(fused, targets, seed=11)
        np.testing.assert_allclose(shuffled[[0, 1]], fused[[1, 0]])
        np.testing.assert_allclose(shuffled[[2, 3]], fused[[3, 2]])
        np.testing.assert_allclose(shuffled.sum(axis=1), 1.0)


class QualityFilteredQuarterGroupsTest(unittest.TestCase):

    def test_filter_removes_duplicates_promotions_irrelevant_text_and_author_dominance(self):
        frame = pd.DataFrame({
            "writer": ["a", "a", "a", "b", "c", "d"],
            "post_date": [1546300800, 1546300900, 1546301000,
                          1546301100, 1546301200, 1546301300],
            "body": [
                "Amazon revenue grows strongly",
                "Amazon revenue grows strongly",
                "Amazon profit margins improve significantly",
                "beautiful sunset over the city",
                "Binary options revenue with huge profit potential",
                "AWS customer demand increased strongly",
            ],
            "class": [2, 2, 2, 2, 2, 2],
        })
        filtered = quality_filter_tweets(
            frame, "amazon", max_author_tweets_per_quarter=1,
            minimum_semantic_tokens=3)
        self.assertEqual(2, len(filtered))
        self.assertEqual({"a", "d"}, set(filtered["writer"]))
        self.assertTrue(filtered["semantic_text"].str.contains("revenue|demand").all())


if __name__ == "__main__":
    unittest.main()
