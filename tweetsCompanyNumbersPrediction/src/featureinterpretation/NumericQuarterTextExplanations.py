"""Leakage-safe explanations for the numeric quarter text models.

The current production branch is a scaled linear model over aggregate text signals, not an
embedding network.  Consequently its exact explanation is ``standardized value * coefficient``.
Important words and NMF topics are attached as two explicitly weaker bridges: the word lexicon is
fitted on past quarters only, while topics summarize held-out documents under a past-only topic
model.  Neither bridge is mislabeled as an additive contribution to the final hybrid decision.
"""

from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from classifier.NumericQuarterTextFeatures import (
    BEAT_PATTERN,
    COMPANY_PATTERNS,
    ESTIMATE_PATTERN,
    FUTURE_PATTERN,
    GUIDANCE_PATTERN,
    METRIC_PATTERNS,
    MISS_PATTERN,
    NEGATIVE_PATTERN,
    NUMBER_PATTERN,
    PERCENT_PATTERN,
    POSITIVE_PATTERN,
    REPORTED_PATTERN,
)
from classifier.PureTextQuarterViews import (
    QuarterStableImportantWords,
    normalize_semantic_text,
)


def _class_linear_parameters(classifier, class_index):
    """Return the OVR parameters for one class, including sklearn's binary convention."""
    if not hasattr(classifier, "coef_"):
        return None, None
    classes = np.asarray(classifier.classes_, dtype=int)
    positions = np.flatnonzero(classes == int(class_index))
    if not len(positions):
        return None, None
    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    intercepts = np.asarray(classifier.intercept_, dtype=np.float64)
    if coefficients.shape[0] == 1 and len(classes) == 2:
        sign = 1.0 if int(class_index) == int(classes[1]) else -1.0
        return sign * coefficients[0], float(sign * intercepts[0])
    position = int(positions[0])
    return coefficients[position], float(intercepts[position])


def linear_class_feature_contributions(fitted_model, row_index, class_index, top_n=None):
    """Exact per-feature contributions to one numeric branch OVR decision score."""
    coefficients, intercept = _class_linear_parameters(
        fitted_model.classifier, class_index)
    if coefficients is None:
        return {"class": int(class_index), "intercept": None, "features": []}
    standardized = np.asarray(
        fitted_model.standardized_evaluation_features[row_index], dtype=np.float64)
    raw = np.asarray(fitted_model.raw_evaluation_features[row_index], dtype=np.float64)
    contributions = standardized * coefficients
    records = [
        {
            "feature": str(name),
            "raw_value": float(raw[index]),
            "standardized_value": float(standardized[index]),
            "coefficient": float(coefficients[index]),
            "signed_contribution": float(contributions[index]),
            "absolute_contribution": float(abs(contributions[index])),
        }
        for index, name in enumerate(fitted_model.feature_names)
    ]
    records.sort(key=lambda value: value["absolute_contribution"], reverse=True)
    if top_n is not None:
        records = records[:int(top_n)]
    return {
        "class": int(class_index),
        "intercept": float(intercept),
        "decision_score": float(intercept + contributions.sum()),
        "features": records,
    }


def aggregate_linear_explanations(explanations, top_n=15):
    """Average exact contributions from repeated seeds without hiding sign instability."""
    by_feature = defaultdict(list)
    intercepts, scores = [], []
    for explanation in explanations:
        if explanation["intercept"] is not None:
            intercepts.append(float(explanation["intercept"]))
            scores.append(float(explanation["decision_score"]))
        for value in explanation["features"]:
            by_feature[value["feature"]].append(value)
    records = []
    for name, values in by_feature.items():
        signed = np.asarray(
            [value["signed_contribution"] for value in values], dtype=np.float64)
        records.append({
            "feature": name,
            "raw_value_mean": float(np.mean([value["raw_value"] for value in values])),
            "standardized_value_mean": float(np.mean([
                value["standardized_value"] for value in values])),
            "coefficient_mean": float(np.mean([
                value["coefficient"] for value in values])),
            "signed_contribution_mean": float(signed.mean()),
            "absolute_contribution_mean": float(np.abs(signed).mean()),
            "sign_agreement": float(abs(np.sign(signed).mean())),
            "seed_count": int(len(values)),
        })
    records.sort(key=lambda value: value["absolute_contribution_mean"], reverse=True)
    return {
        "intercept_mean": float(np.mean(intercepts)) if intercepts else None,
        "decision_score_mean": float(np.mean(scores)) if scores else None,
        "features": records[:int(top_n)],
    }


def _feature_families(feature_name):
    name = feature_name.lower()
    families = []
    for family in ("reported", "estimate", "guidance", "beat", "miss", "future"):
        if family in name:
            families.append(family)
    if "positive" in name or "class_2" in name or "class_3" in name:
        families.append("positive")
    if "negative" in name or "class_0" in name:
        families.append("negative")
    if "percent" in name or "change_scaled" in name:
        families.append("percent")
    if "level" in name or "metric" in name or "change_scaled" in name:
        families.append("metric")
    if "company__" in name:
        families.append("company")
    if "total_tweets" in name:
        families.append("volume")
    return tuple(dict.fromkeys(families)) or ("other",)


_FAMILY_PATTERNS = {
    "reported": REPORTED_PATTERN,
    "estimate": ESTIMATE_PATTERN,
    "guidance": GUIDANCE_PATTERN,
    "beat": BEAT_PATTERN,
    "miss": MISS_PATTERN,
    "future": FUTURE_PATTERN,
    "positive": POSITIVE_PATTERN,
    "negative": NEGATIVE_PATTERN,
}


def _family_terms(documents, company):
    counters = defaultdict(Counter)
    for document in documents:
        text = "" if document is None else str(document)
        for family, pattern in _FAMILY_PATTERNS.items():
            counters[family].update(
                " ".join(match.group(0).lower().split())
                for match in pattern.finditer(text)
            )
        counters["company"].update(
            " ".join(match.group(0).lower().split())
            for match in COMPANY_PATTERNS[company].finditer(text)
        )
        counters["metric"].update(
            " ".join(match.group(0).lower().split())
            for match in METRIC_PATTERNS[company].finditer(text)
        )
        counters["percent"]["<percentage_value>"] += len(PERCENT_PATTERN.findall(text))
        counters["metric"]["<numeric_value>"] += len(NUMBER_PATTERN.findall(text))
    counters["volume"]["<tweet_volume>"] = len(documents)
    return counters


def model_linked_cue_words(documents, company, feature_contributions, top_n=15):
    """Bridge exact aggregate-feature contributions to the cue words that form them.

    The allocation within a feature family is descriptive.  It is not an exact token-level
    decomposition because medians, quantiles and quarter aggregation are nonlinear.
    """
    family_signed, family_absolute = defaultdict(float), defaultdict(float)
    for value in feature_contributions:
        families = _feature_families(value["feature"])
        share = 1.0 / len(families)
        signed = float(value.get(
            "signed_contribution_mean", value.get("signed_contribution", 0.0)))
        absolute = float(value.get(
            "absolute_contribution_mean", value.get("absolute_contribution", abs(signed))))
        for family in families:
            family_signed[family] += share * signed
            family_absolute[family] += share * absolute
    terms = _family_terms(documents, company)
    records = []
    for family, absolute_weight in family_absolute.items():
        family_counts = terms.get(family, Counter())
        count_sum = sum(family_counts.values())
        if not count_sum:
            continue
        for term, count in family_counts.items():
            fraction = float(count) / count_sum
            records.append({
                "term": str(term),
                "evidence_family": family,
                "occurrences": int(count),
                "signed_weight": float(family_signed[family] * fraction),
                "absolute_weight": float(absolute_weight * fraction),
            })
    records.sort(key=lambda value: value["absolute_weight"], reverse=True)
    return records[:int(top_n)]


def balanced_documents(documents_by_quarter, quarters, max_per_quarter=250):
    """Deterministically cap each quarter so prolific periods cannot dominate topics."""
    documents, document_quarters = [], []
    for quarter in quarters:
        values = list(documents_by_quarter.get(quarter, []))
        if len(values) > max_per_quarter:
            indexes = np.linspace(
                0, len(values) - 1, int(max_per_quarter), dtype=int)
            values = [values[index] for index in indexes]
        for value in values:
            normalized = normalize_semantic_text(value)
            if normalized:
                documents.append(normalized)
                document_quarters.append(quarter)
    return documents, document_quarters


def fit_past_only_important_words(documents_by_quarter, labels_by_quarter,
                                  train_quarters, max_per_quarter=250):
    documents, quarters = balanced_documents(
        documents_by_quarter, train_quarters, max_per_quarter)
    labels = [int(labels_by_quarter[quarter]) for quarter in quarters]
    if len(documents) < 2 or len(set(labels)) < 2:
        return None
    try:
        return QuarterStableImportantWords(
            max_features=20000,
            minimum_global_quarters=2,
            minimum_class_quarters=2,
        ).fit(documents, labels, quarters)
    except ValueError:
        return None


def heldout_important_words(model, documents, class_index, top_n=15):
    """Rank past-learned class words that are actually present in held-out text."""
    if model is None or not model.available[int(class_index)]:
        return []
    normalized = [normalize_semantic_text(value) for value in documents]
    normalized = [value for value in normalized if value]
    if not normalized:
        return []
    matrix = model.vectorizer.transform(normalized)
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    scores = model.class_log_odds[int(class_index)]
    relevance = scores * document_frequency / len(normalized)
    indexes = np.flatnonzero(relevance > 0.0)
    indexes = indexes[np.argsort(relevance[indexes])[::-1]][:int(top_n)]
    names = model.vectorizer.get_feature_names_out()
    return [
        {
            "term": str(names[index]),
            "past_only_stable_log_odds": float(scores[index]),
            "heldout_document_fraction": float(document_frequency[index] / len(normalized)),
            "relevance": float(relevance[index]),
        }
        for index in indexes
    ]


class PastOnlyNmfTopics:
    """Small deterministic topic model fitted exclusively on earlier quarters."""

    def __init__(self, topic_count=6, max_features=2000, seed=1337):
        self.topic_count = int(topic_count)
        self.max_features = int(max_features)
        self.seed = int(seed)
        self.vectorizer = None
        self.model = None

    def fit(self, documents):
        documents = [value for value in documents if value]
        if len(documents) < 2:
            return self
        configurations = ((2, 0.98), (1, 1.0))
        matrix = None
        for min_df, max_df in configurations:
            vectorizer = TfidfVectorizer(
                lowercase=False,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=min_df,
                max_df=max_df,
                max_features=self.max_features,
                sublinear_tf=True,
                dtype=np.float32,
            )
            try:
                candidate = vectorizer.fit_transform(documents)
            except ValueError:
                continue
            if candidate.shape[1]:
                self.vectorizer, matrix = vectorizer, candidate
                break
        if matrix is None:
            return self
        components = min(self.topic_count, matrix.shape[0], matrix.shape[1])
        self.model = NMF(
            n_components=max(1, components),
            init="nndsvda",
            random_state=self.seed,
            max_iter=500,
        ).fit(matrix)
        return self

    @property
    def available(self):
        return self.vectorizer is not None and self.model is not None

    def catalog(self, words_per_topic=10):
        if not self.available:
            return []
        names = self.vectorizer.get_feature_names_out()
        result = []
        for topic_index, weights in enumerate(self.model.components_):
            indexes = np.argsort(weights)[::-1][:int(words_per_topic)]
            result.append({
                "topic_id": int(topic_index),
                "top_words": [str(names[index]) for index in indexes],
                "word_weights": [float(weights[index]) for index in indexes],
            })
        return result

    def describe(self, documents, important_words=(), cue_words=(), top_n=3):
        if not self.available:
            return []
        normalized = [normalize_semantic_text(value) for value in documents]
        normalized = [value for value in normalized if value]
        if not normalized:
            return []
        weights = self.model.transform(self.vectorizer.transform(normalized))
        prevalence = weights.mean(axis=0)
        total = max(float(prevalence.sum()), 1e-12)
        important = {value["term"] for value in important_words}
        cues = {value["term"] for value in cue_words if not value["term"].startswith("<")}
        catalog = self.catalog()
        result = []
        for topic in catalog:
            words = set(topic["top_words"])
            result.append({
                **topic,
                "heldout_weight": float(prevalence[topic["topic_id"]]),
                "heldout_share": float(prevalence[topic["topic_id"]] / total),
                "important_word_overlap": sorted(words.intersection(important)),
                "model_cue_overlap": sorted(words.intersection(cues)),
            })
        result.sort(key=lambda value: value["heldout_weight"], reverse=True)
        return result[:int(top_n)]
