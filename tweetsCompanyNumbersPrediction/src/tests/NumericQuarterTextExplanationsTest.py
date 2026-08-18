import unittest
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression

from extractNumericQuarterTopicsAndImportantWords import _forbidden_output_key_audit
from featureinterpretation.NumericQuarterTextExplanations import (
    PastOnlyNmfTopics,
    fit_past_only_important_words,
    heldout_important_words,
    linear_class_feature_contributions,
    model_linked_cue_words,
)


class NumericQuarterTextExplanationsTest(unittest.TestCase):

    def test_linear_feature_contributions_reconstruct_ovr_decision_score(self):
        features = np.asarray([
            [-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [0.5, 0.5],
            [1.0, -0.5], [2.0, 0.0], [3.0, 1.0], [4.0, -1.0],
        ])
        labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
        classifier = LogisticRegression(
            solver="liblinear", multi_class="ovr", random_state=7).fit(features, labels)
        fitted = SimpleNamespace(
            classifier=classifier,
            raw_evaluation_features=features,
            standardized_evaluation_features=features,
            feature_names=("reported_tweet_fraction", "signed_percent_mean_scaled"),
        )
        explanation = linear_class_feature_contributions(fitted, 6, 3)
        expected = float(classifier.decision_function(features[6:7])[0, 3])
        self.assertAlmostEqual(expected, explanation["decision_score"], places=10)
        reconstructed = explanation["intercept"] + sum(
            value["signed_contribution"] for value in explanation["features"])
        self.assertAlmostEqual(expected, reconstructed, places=10)

    def test_model_linked_words_expose_cues_without_raw_sentences(self):
        documents = [
            "Amazon reported revenue up 22 percent and beat estimates.",
            "AMZN revenue guidance may increase by 10%.",
        ]
        features = [
            {"feature": "all__reported_tweet_fraction",
             "signed_contribution_mean": 0.8, "absolute_contribution_mean": 0.8},
            {"feature": "all__signed_percent_positive_fraction",
             "signed_contribution_mean": 0.4, "absolute_contribution_mean": 0.4},
        ]
        words = model_linked_cue_words(documents, "amazon", features, top_n=20)
        terms = {value["term"] for value in words}
        self.assertIn("reported", terms)
        self.assertTrue(terms.intersection({"up", "increase", "beat"}))
        self.assertNotIn(documents[0], terms)

    def test_past_only_words_are_fitted_before_heldout_text(self):
        documents = {
            "2015Q1": ["revenue decline weak", "sales decline lower"],
            "2015Q2": ["revenue surge record", "sales surge higher"],
            "2016Q1": ["revenue decline falling", "sales decline weak"],
            "2016Q2": ["revenue surge growth", "sales surge record"],
        }
        labels = {"2015Q1": 0, "2015Q2": 3, "2016Q1": 0, "2016Q2": 3}
        lexicon = fit_past_only_important_words(
            documents, labels, sorted(documents), max_per_quarter=10)
        words = heldout_important_words(
            lexicon, ["Amazon revenue surge reached a record"], 3, top_n=10)
        self.assertIn("surge", {value["term"] for value in words})

    def test_topics_return_only_aggregate_descriptors(self):
        documents = [
            "revenue cloud growth enterprise demand",
            "cloud sales growth customer demand",
            "delivery production vehicles factory output",
            "vehicle deliveries production factory",
        ]
        topics = PastOnlyNmfTopics(topic_count=2, seed=3).fit(documents)
        descriptions = topics.describe(
            ["cloud revenue growth demand"], top_n=2)
        self.assertTrue(descriptions)
        self.assertIn("top_words", descriptions[0])
        self.assertNotIn("document", descriptions[0])
        self.assertNotIn("body", descriptions[0])

    def test_privacy_audit_rejects_raw_content_keys(self):
        _forbidden_output_key_audit({"topics": [{"top_words": ["growth"]}]})
        with self.assertRaises(AssertionError):
            _forbidden_output_key_audit({"tweet_id": 123})


if __name__ == "__main__":
    unittest.main()
