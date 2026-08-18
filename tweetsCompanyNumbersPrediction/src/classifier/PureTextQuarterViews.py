"""Pure-text group classifiers and quarter aggregation utilities."""

import re

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


NUM_CLASSES = 4
STYLE_FEATURE_NAMES = (
    "vader_negative",
    "vader_neutral",
    "vader_positive",
    "vader_compound",
    "log_character_count",
    "log_token_count",
    "unique_token_fraction",
    "uppercase_character_fraction",
    "digit_character_fraction",
    "exclamation_per_character",
    "question_per_character",
    "url_per_tweet",
    "cashtag_per_tweet",
    "percent_symbol_per_tweet",
    "future_language_per_tweet",
    "uncertainty_language_per_tweet",
    "log_unique_authors",
    "top_author_fraction",
    "log_group_time_span_seconds",
)

FINANCE_EVENT_TERMS = frozenset({
    "earnings", "eps", "revenue", "sales", "sold", "deliver", "delivered", "deliveries",
    "shipment", "shipments", "production", "produce", "units", "orders", "demand",
    "profit", "profits", "profitable", "loss", "losses", "margin", "margins", "cash",
    "growth", "grow", "grew", "decline", "declined", "increase", "increased", "decrease",
    "decreased", "rise", "rose", "fall", "fell", "record", "beat", "beats", "miss",
    "missed", "estimate", "estimates", "forecast", "guidance", "outlook", "quarter",
    "quarterly", "price", "prices", "cost", "costs", "market", "share", "launch",
})

SEMANTIC_NOISE_TERMS = frozenset({
    "http", "https", "www", "com", "amp", "utm", "utm_source", "utm_medium",
    "utm_campaign", "source", "campaign", "signup", "subscribe", "subscriber",
    "subscribers", "alert", "alerts", "click", "link", "website", "free", "video",
    "newsletter", "breaking", "stockmarket", "stocks", "nasdaq", "nyse", "portfolio",
})

MONTH_AND_DAY_TERMS = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
})


def normalize_semantic_text(text):
    """Remove period/source shortcuts while preserving ordinary semantic words."""
    text = "" if text is None else str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\S+\.(?:com|net|org|io|co|news)\b\S*", " ", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"\$[A-Za-z]{1,6}\b", " ", text)
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\b\d+(?:[./:-]\d+)+\b", " ", text)
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    tokens = [
        token for token in tokens
        if token not in SEMANTIC_NOISE_TERMS and token not in MONTH_AND_DAY_TERMS
    ]
    return " ".join(tokens)


def finance_relevant_semantic_text(bodies):
    """Keep finance/event-bearing tweets, falling back to the normalized whole group."""
    normalized = [normalize_semantic_text(body) for body in bodies]
    relevant = [
        text for text in normalized
        if FINANCE_EVENT_TERMS.intersection(text.split())
    ]
    return " <SEP> ".join(relevant if relevant else normalized)


def word_vectorizer(max_features=50000):
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def character_vectorizer(max_features=60000):
    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def semantic_word_vectorizer(max_features=50000):
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )


class ConstantClassifier:
    """Probability-compatible classifier for folds containing only one past class."""

    def __init__(self, label):
        self.classes_ = np.asarray([int(label)], dtype=np.int64)

    def predict_proba(self, features):
        return np.ones((features.shape[0], 1), dtype=np.float64)


def fit_logistic_classifier(features, labels, regularization, seed):
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(labels)
    if len(unique) == 1:
        return ConstantClassifier(unique[0])
    classifier = LogisticRegression(
        C=float(regularization),
        class_weight="balanced",
        solver="liblinear",
        multi_class="ovr",
        max_iter=400,
        random_state=seed,
    )
    classifier.fit(features, labels)
    return classifier


def mapped_probabilities(classifier, features, num_classes=NUM_CLASSES):
    probabilities = classifier.predict_proba(features)
    mapped = np.zeros((features.shape[0], num_classes), dtype=np.float64)
    mapped[:, classifier.classes_.astype(int)] = probabilities
    denominator = mapped.sum(axis=1, keepdims=True)
    return mapped / np.maximum(denominator, 1e-12)


class SparseOrdinalClassifier:
    """Three past-only sparse classifiers for the ordered four-class target."""

    def __init__(self, regularization=1.0, seed=1337):
        self.regularization = float(regularization)
        self.seed = int(seed)
        self.threshold_models = []

    def fit(self, features, labels):
        labels = np.asarray(labels, dtype=np.int64)
        self.threshold_models = [
            fit_logistic_classifier(
                features, (labels > threshold).astype(np.int64),
                self.regularization, self.seed + threshold)
            for threshold in range(NUM_CLASSES - 1)
        ]
        return self

    @staticmethod
    def _positive_probability(classifier, features):
        probabilities = classifier.predict_proba(features)
        positive = np.zeros(features.shape[0], dtype=np.float64)
        matching = np.flatnonzero(classifier.classes_.astype(int) == 1)
        if len(matching):
            positive = probabilities[:, matching[0]]
        return positive

    def threshold_probabilities(self, features):
        values = np.column_stack([
            self._positive_probability(model, features)
            for model in self.threshold_models
        ])
        # P(y>0) >= P(y>1) >= P(y>2) must hold for valid ordinal probabilities.
        return np.minimum.accumulate(values, axis=1)

    def predict_proba(self, features):
        thresholds = self.threshold_probabilities(features)
        probabilities = np.column_stack((
            1.0 - thresholds[:, 0],
            thresholds[:, 0] - thresholds[:, 1],
            thresholds[:, 1] - thresholds[:, 2],
            thresholds[:, 2],
        ))
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)

    def predict_direction_proba(self, features):
        positive = self.threshold_probabilities(features)[:, 0]
        return np.column_stack((1.0 - positive, positive))


class ImportantWordLogOdds:
    """Past-only class log-odds lexicon resembling the important-word analysis."""

    def __init__(self, max_features=40000, smoothing=1.0):
        self.vectorizer = CountVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.995,
            max_features=max_features,
            binary=True,
            dtype=np.float32,
        )
        self.smoothing = float(smoothing)
        self.class_log_odds = None
        self.available = None

    def fit(self, texts, labels):
        features = self.vectorizer.fit_transform(texts)
        labels = np.asarray(labels, dtype=np.int64)
        vocabulary_size = features.shape[1]
        self.class_log_odds = np.zeros((NUM_CLASSES, vocabulary_size), dtype=np.float64)
        self.available = np.zeros(NUM_CLASSES, dtype=bool)
        for class_index in range(NUM_CLASSES):
            class_mask = labels == class_index
            if not class_mask.any():
                continue
            self.available[class_index] = True
            class_counts = np.asarray(features[class_mask].sum(axis=0)).ravel()
            rest_counts = np.asarray(features[~class_mask].sum(axis=0)).ravel()
            class_probability = (
                class_counts + self.smoothing
            ) / (class_counts.sum() + self.smoothing * vocabulary_size)
            rest_probability = (
                rest_counts + self.smoothing
            ) / (rest_counts.sum() + self.smoothing * vocabulary_size)
            self.class_log_odds[class_index] = np.log(class_probability / rest_probability)
        return self

    def decision_function(self, texts):
        features = self.vectorizer.transform(texts)
        token_counts = np.maximum(np.asarray(features.sum(axis=1)).ravel(), 1.0)
        scores = np.asarray(features @ self.class_log_odds.T)
        scores /= token_counts[:, None]
        scores[:, ~self.available] = -30.0
        return scores

    def predict_proba(self, texts, temperature=1.0):
        return softmax(self.decision_function(texts) / float(temperature), axis=1)

    def top_words(self, count=25):
        names = self.vectorizer.get_feature_names_out()
        result = {}
        for class_index in range(NUM_CLASSES):
            if not self.available[class_index]:
                result[str(class_index)] = []
                continue
            indexes = np.argsort(self.class_log_odds[class_index])[-count:][::-1]
            result[str(class_index)] = [
                {
                    "token": str(names[index]),
                    "log_odds": float(self.class_log_odds[class_index, index]),
                }
                for index in indexes
            ]
        return result


class QuarterStableImportantWords:
    """Class lexicon based on recurrence across past quarters, not raw group frequency."""

    def __init__(self, max_features=40000, smoothing=0.25,
                 minimum_global_quarters=2, minimum_class_quarters=2):
        self.vectorizer = CountVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=max_features,
            binary=True,
            stop_words="english",
            dtype=np.float32,
        )
        self.smoothing = float(smoothing)
        self.minimum_global_quarters = int(minimum_global_quarters)
        self.minimum_class_quarters = int(minimum_class_quarters)
        self.class_log_odds = None
        self.available = None

    def fit(self, texts, labels, quarters):
        features = self.vectorizer.fit_transform(texts)
        labels = np.asarray(labels, dtype=np.int64)
        quarters = np.asarray(quarters)
        unique_quarters = sorted(set(quarters))
        quarter_prevalence, quarter_labels = [], []
        for quarter in unique_quarters:
            mask = quarters == quarter
            values = np.unique(labels[mask])
            if len(values) != 1:
                raise ValueError("Quarter %s contains multiple labels" % quarter)
            quarter_prevalence.append(np.asarray(features[mask].mean(axis=0)).ravel())
            quarter_labels.append(int(values[0]))
        prevalence = np.asarray(quarter_prevalence, dtype=np.float64)
        quarter_labels = np.asarray(quarter_labels, dtype=np.int64)
        present = prevalence > 0.0
        global_recurrence = present.sum(axis=0)
        self.class_log_odds = np.zeros((NUM_CLASSES, features.shape[1]), dtype=np.float64)
        self.available = np.zeros(NUM_CLASSES, dtype=bool)
        for class_index in range(NUM_CLASSES):
            class_mask = quarter_labels == class_index
            rest_mask = ~class_mask
            if not class_mask.any() or not rest_mask.any():
                continue
            self.available[class_index] = True
            class_recurrence = present[class_mask].sum(axis=0)
            stable = (
                (global_recurrence >= self.minimum_global_quarters)
                & (class_recurrence >= self.minimum_class_quarters)
            )
            class_probability = (
                prevalence[class_mask].sum(axis=0) + self.smoothing
            ) / (class_mask.sum() + 2.0 * self.smoothing)
            rest_probability = (
                prevalence[rest_mask].sum(axis=0) + self.smoothing
            ) / (rest_mask.sum() + 2.0 * self.smoothing)
            class_logit = np.log(class_probability / np.maximum(1.0 - class_probability, 1e-8))
            rest_logit = np.log(rest_probability / np.maximum(1.0 - rest_probability, 1e-8))
            self.class_log_odds[class_index, stable] = (
                class_logit[stable] - rest_logit[stable])
        return self

    def decision_function(self, texts):
        features = self.vectorizer.transform(texts)
        token_counts = np.maximum(np.asarray(features.sum(axis=1)).ravel(), 1.0)
        scores = np.asarray(features @ self.class_log_odds.T) / token_counts[:, None]
        scores[:, ~self.available] = -30.0
        return scores

    def predict_proba(self, texts, temperature=1.0):
        return softmax(self.decision_function(texts) / float(temperature), axis=1)

    def top_words(self, count=25):
        names = self.vectorizer.get_feature_names_out()
        result = {}
        for class_index in range(NUM_CLASSES):
            positive = np.flatnonzero(self.class_log_odds[class_index] > 0)
            indexes = positive[np.argsort(
                self.class_log_odds[class_index, positive])[-count:][::-1]]
            result[str(class_index)] = [
                {
                    "token": str(names[index]),
                    "stable_log_odds": float(self.class_log_odds[class_index, index]),
                }
                for index in indexes
            ]
        return result


def group_style_features(bodies, authors, post_dates, analyzer=None):
    """Extract non-embedding linguistic, sentiment, author and timing features."""
    analyzer = analyzer or SentimentIntensityAnalyzer()
    bodies = ["" if body is None else str(body) for body in bodies]
    text = " ".join(bodies)
    sentiment = analyzer.polarity_scores(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    characters = max(len(text), 1)
    tweets = max(len(bodies), 1)
    alphabetic = [character for character in text if character.isalpha()]
    uppercase_fraction = (
        sum(character.isupper() for character in alphabetic) / max(len(alphabetic), 1))
    authors = ["<missing>" if author is None else str(author) for author in authors]
    _, author_counts = np.unique(authors, return_counts=True)
    top_author_fraction = float(author_counts.max() / max(author_counts.sum(), 1))
    post_dates = np.asarray(post_dates, dtype=np.float64)
    time_span = float(post_dates.max() - post_dates.min()) if len(post_dates) else 0.0
    future_terms = re.findall(
        r"\b(?:will|expect|expects|expected|forecast|guidance|future|launch|next|upcoming)\b",
        text,
        flags=re.IGNORECASE,
    )
    uncertainty_terms = re.findall(
        r"\b(?:may|might|could|risk|uncertain|uncertainty|possibly|perhaps|rumou?r)\b",
        text,
        flags=re.IGNORECASE,
    )
    return np.asarray([
        sentiment["neg"],
        sentiment["neu"],
        sentiment["pos"],
        sentiment["compound"],
        np.log1p(len(text)),
        np.log1p(len(tokens)),
        len(set(tokens)) / max(len(tokens), 1),
        uppercase_fraction,
        sum(character.isdigit() for character in text) / characters,
        text.count("!") / characters,
        text.count("?") / characters,
        len(re.findall(r"https?://|www\.", text, flags=re.IGNORECASE)) / tweets,
        len(re.findall(r"\$[A-Za-z]{1,6}\b", text)) / tweets,
        text.count("%") / tweets,
        len(future_terms) / tweets,
        len(uncertainty_terms) / tweets,
        np.log1p(len(author_counts)),
        top_author_fraction,
        np.log1p(max(time_span, 0.0)),
    ], dtype=np.float32)


def style_classifier(regularization, seed):
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=float(regularization),
            class_weight="balanced",
            solver="liblinear",
            multi_class="ovr",
            max_iter=400,
            random_state=seed,
        )),
    ])


def fit_style_classifier(features, labels, regularization, seed):
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(labels)
    if len(unique) == 1:
        return ConstantClassifier(unique[0])
    return style_classifier(regularization, seed).fit(features, labels)


def style_probabilities(classifier, features):
    probabilities = classifier.predict_proba(features)
    classes = (classifier.classes_ if hasattr(classifier, "classes_")
               else classifier.named_steps["classifier"].classes_)
    mapped = np.zeros((len(features), NUM_CLASSES), dtype=np.float64)
    mapped[:, np.asarray(classes, dtype=int)] = probabilities
    return mapped / np.maximum(mapped.sum(axis=1, keepdims=True), 1e-12)


def aggregate_quarter_probabilities(group_quarters, probabilities, quarters):
    """Reduce any number of group predictions to one probability vector per quarter."""
    group_quarters = np.asarray(group_quarters)
    result = []
    for quarter in quarters:
        mask = group_quarters == quarter
        if not mask.any():
            raise ValueError("No group probabilities found for quarter %s" % quarter)
        aggregated = probabilities[mask].mean(axis=0)
        result.append(aggregated / max(aggregated.sum(), 1e-12))
    return np.asarray(result, dtype=np.float64)


TEMPORAL_AGGREGATION_MODES = (
    "mean",
    "vote",
    "geometric",
    "early_half_mean",
    "late_half_mean",
    "last_third_mean",
    "confidence_top_quartile",
    "late_half_vote",
)


def temperature_scale_probabilities(probabilities, temperature):
    log_values = np.log(np.clip(probabilities, 1e-8, 1.0)) / float(temperature)
    return softmax(log_values, axis=1)


def _vote_distribution(probabilities):
    counts = np.bincount(probabilities.argmax(axis=1), minlength=NUM_CLASSES).astype(np.float64)
    counts += 1e-3
    return counts / counts.sum()


def aggregate_temporal_quarter_probabilities(group_quarters, group_timestamps, probabilities,
                                             quarters, mode="mean", temperature=1.0):
    """Aggregate group evidence using only within-current-quarter time and confidence."""
    if mode not in TEMPORAL_AGGREGATION_MODES:
        raise ValueError("Unknown temporal aggregation mode %s" % mode)
    group_quarters = np.asarray(group_quarters)
    group_timestamps = np.asarray(group_timestamps, dtype=np.float64)
    probabilities = temperature_scale_probabilities(probabilities, temperature)
    result = []
    for quarter in quarters:
        indexes = np.flatnonzero(group_quarters == quarter)
        if not len(indexes):
            raise ValueError("No group probabilities found for quarter %s" % quarter)
        indexes = indexes[np.argsort(group_timestamps[indexes], kind="stable")]
        if mode == "early_half_mean":
            indexes = indexes[:max(1, int(np.ceil(len(indexes) / 2.0)))]
        elif mode in {"late_half_mean", "late_half_vote"}:
            indexes = indexes[len(indexes) // 2:]
        elif mode == "last_third_mean":
            indexes = indexes[max(0, len(indexes) - int(np.ceil(len(indexes) / 3.0))):]
        elif mode == "confidence_top_quartile":
            count = max(1, int(np.ceil(len(indexes) / 4.0)))
            confidence_order = np.argsort(probabilities[indexes].max(axis=1))
            indexes = indexes[confidence_order[-count:]]
        selected = probabilities[indexes]
        if mode in {"vote", "late_half_vote"}:
            aggregated = _vote_distribution(selected)
        elif mode == "geometric":
            aggregated = softmax(np.log(np.clip(selected, 1e-8, 1.0)).mean(axis=0))
        else:
            aggregated = selected.mean(axis=0)
            aggregated /= max(aggregated.sum(), 1e-12)
        result.append(aggregated)
    return np.asarray(result, dtype=np.float64)
