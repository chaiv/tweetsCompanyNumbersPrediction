"""Target-aware but label-independent numeric signals extracted from tweet text."""

import re

import numpy as np


COMPANY_PATTERNS = {
    "amazon": re.compile(r"\b(?:amazon|amzn)\b", re.IGNORECASE),
    "apple": re.compile(r"\b(?:apple|aapl)\b", re.IGNORECASE),
    "tesla": re.compile(r"\b(?:tesla|tsla)\b", re.IGNORECASE),
}
METRIC_PATTERNS = {
    "amazon": re.compile(r"\b(?:revenue|net sales|aws sales)\b", re.IGNORECASE),
    "apple": re.compile(r"\b(?:eps|earnings per share)\b", re.IGNORECASE),
    "tesla": re.compile(
        r"\b(?:deliver(?:y|ies|ed)|vehicle deliveries|production)\b",
        re.IGNORECASE,
    ),
}
LEVEL_METRIC_PATTERNS = {
    "amazon": METRIC_PATTERNS["amazon"],
    "apple": METRIC_PATTERNS["apple"],
    "tesla": re.compile(
        r"\b(?:deliver(?:y|ies|ed)|vehicle deliveries)\b", re.IGNORECASE),
}
PERCENT_PATTERN = re.compile(
    r"(?P<value>[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:%|percent\b)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"(?P<currency>\$)?\s*(?P<value>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<unit>billion|million|bn|mm|[bk])?\b",
    re.IGNORECASE,
)
POSITIVE_PATTERN = re.compile(
    r"\b(?:up|increase[ds]?|increasing|growth|grew|grow|rise|rises|rose|rising|"
    r"gain(?:ed|s)?|higher|beat|beats|beating|exceed(?:ed|s)?|surge[ds]?)\b",
    re.IGNORECASE,
)
NEGATIVE_PATTERN = re.compile(
    r"\b(?:down|decrease[ds]?|decreasing|decline[ds]?|declining|fell|fall(?:ing)?|"
    r"drop(?:ped|s)?|lower|miss|missed|misses|slump(?:ed|s)?)\b",
    re.IGNORECASE,
)
REPORTED_PATTERN = re.compile(
    r"\b(?:report(?:ed|s|ing)?|actual|announc(?:ed|es)|posted)\b", re.IGNORECASE)
ESTIMATE_PATTERN = re.compile(
    r"\b(?:estimate[ds]?|expect(?:ed|s)?|consensus|forecast(?:ed|s)?)\b",
    re.IGNORECASE,
)
GUIDANCE_PATTERN = re.compile(r"\b(?:guidance|outlook)\b", re.IGNORECASE)
BEAT_PATTERN = re.compile(r"\b(?:beat|beats|beating|exceed(?:ed|s)?)\b", re.IGNORECASE)
MISS_PATTERN = re.compile(r"\b(?:miss|missed|misses|below estimates?)\b", re.IGNORECASE)
FUTURE_PATTERN = re.compile(
    r"\b(?:will|may|might|could|next quarter|upcoming)\b", re.IGNORECASE)

NUMERIC_FEATURE_NAMES = (
    "log_total_tweets",
    "log_metric_tweets",
    "metric_tweet_fraction",
    "log_percent_mentions",
    "percent_mentions_per_metric_tweet",
    "signed_percent_fraction",
    "signed_percent_mean_scaled",
    "signed_percent_median_scaled",
    "signed_percent_q25_scaled",
    "signed_percent_q75_scaled",
    "signed_percent_positive_fraction",
    "signed_percent_negative_fraction",
    "percent_class_0_fraction",
    "percent_class_1_fraction",
    "percent_class_2_fraction",
    "percent_class_3_fraction",
    "reported_tweet_fraction",
    "estimate_tweet_fraction",
    "guidance_tweet_fraction",
    "beat_tweet_fraction",
    "miss_tweet_fraction",
    "future_tweet_fraction",
    "log_level_mentions",
    "log_level_median",
    "log_level_mode",
    "log_level_q75",
    "log_level_q90",
    "log_level_iqr",
)


def contains_company_and_metric(text, company):
    text = "" if text is None else str(text)
    return bool(COMPANY_PATTERNS[company].search(text)) and bool(
        METRIC_PATTERNS[company].search(text))


def _window(text, start, end, radius=100):
    return text[max(0, start - radius):min(len(text), end + radius)]


def _nearest_direction(window, match_start, explicit_value):
    if explicit_value.startswith("-"):
        return -1.0
    if explicit_value.startswith("+"):
        return 1.0
    directions = []
    for pattern, sign in ((POSITIVE_PATTERN, 1.0), (NEGATIVE_PATTERN, -1.0)):
        for match in pattern.finditer(window):
            center = (match.start() + match.end()) / 2.0
            directions.append((abs(center - match_start), sign))
    return min(directions)[1] if directions else None


def _signed_percentages(text, metric_pattern):
    result, all_count = [], 0
    for match in PERCENT_PATTERN.finditer(text):
        window_start = max(0, match.start() - 100)
        window = _window(text, match.start(), match.end(), radius=100)
        if not metric_pattern.search(window):
            continue
        value_text = match.group("value")
        value = abs(float(value_text.replace(",", "")))
        if value > 300.0:
            continue
        all_count += 1
        direction = _nearest_direction(
            window, match.start() - window_start, value_text)
        if direction is not None:
            result.append(direction * value)
    return result, all_count


def _numeric_value(match):
    value = float(match.group("value").replace(",", ""))
    unit = (match.group("unit") or "").lower()
    return value, unit


def _normalized_level(match, company, window):
    suffix = window[match.end():match.end() + 2]
    if suffix.lstrip().startswith("%"):
        return None
    value, unit = _numeric_value(match)
    if 1900 <= value <= 2100 and not unit:
        return None
    if company == "amazon":
        if unit in {"billion", "bn", "b"}:
            value_in_billions = value
        elif unit in {"million", "mm"}:
            value_in_billions = value / 1000.0
        else:
            return None
        return value_in_billions if 0.1 <= value_in_billions <= 1000.0 else None
    if company == "apple":
        if unit or (not match.group("currency") and "." not in match.group("value")):
            return None
        return value if 0.01 <= value <= 50.0 else None
    if unit == "k":
        value *= 1000.0
    elif unit in {"million", "mm"}:
        value *= 1000000.0
    elif unit in {"billion", "bn", "b"}:
        value *= 1000000000.0
    return value if 1000.0 <= value <= 200000.0 else None


def _level_mentions(text, company, metric_pattern):
    result, used_positions = [], set()
    level_metric_pattern = LEVEL_METRIC_PATTERNS[company]
    for metric_match in level_metric_pattern.finditer(text):
        window_start = max(0, metric_match.start() - 70)
        window_end = min(len(text), metric_match.end() + 70)
        window = text[window_start:window_end]
        local_metric_center = (
            metric_match.start() + metric_match.end()) / 2.0 - window_start
        candidates = []
        for match in NUMBER_PATTERN.finditer(window):
            value = _normalized_level(match, company, window)
            if value is None:
                continue
            global_position = window_start + match.start()
            if global_position in used_positions:
                continue
            center = (match.start() + match.end()) / 2.0
            candidates.append((abs(center - local_metric_center), global_position, value))
        if candidates:
            _, position, value = min(candidates)
            used_positions.add(position)
            result.append(value)
    return result


def _class_for_percent(value):
    if value < 0.0:
        return 0
    if value <= 15.0:
        return 1
    if value <= 30.0:
        return 2
    return 3


def _safe_distribution_counts(values, size):
    if not len(values):
        return np.zeros(size, dtype=np.float64)
    return np.bincount(values, minlength=size).astype(np.float64) / len(values)


def _level_mode(values, company):
    if not len(values):
        return 0.0
    resolution = {"amazon": 0.1, "apple": 0.01, "tesla": 100.0}[company]
    quantized = np.round(np.asarray(values) / resolution) * resolution
    unique, counts = np.unique(quantized, return_counts=True)
    return float(unique[counts.argmax()])


def _matching_fraction(bodies, pattern):
    return float(np.mean([bool(pattern.search(text)) for text in bodies])) if bodies else 0.0


def numeric_quarter_features(metric_bodies, company, total_tweets):
    """Aggregate numeric target-language from one current quarter without labels."""
    bodies = [
        "" if body is None else str(body) for body in metric_bodies
        if contains_company_and_metric(body, company)
    ]
    metric_pattern = METRIC_PATTERNS[company]
    signed_percentages, all_percent_count, levels = [], 0, []
    for text in bodies:
        signed, count = _signed_percentages(text, metric_pattern)
        signed_percentages.extend(signed)
        all_percent_count += count
        levels.extend(_level_mentions(text, company, metric_pattern))

    signed = np.asarray(signed_percentages, dtype=np.float64)
    clipped = np.clip(signed, -200.0, 300.0) / 100.0
    metric_count = max(len(bodies), 1)
    classes = [_class_for_percent(value) for value in signed]
    class_fractions = _safe_distribution_counts(classes, 4)
    level_values = np.asarray(levels, dtype=np.float64)
    log_levels = np.log1p(level_values) if len(level_values) else np.asarray([])
    result = np.asarray([
        np.log1p(total_tweets),
        np.log1p(len(bodies)),
        len(bodies) / max(total_tweets, 1),
        np.log1p(all_percent_count),
        all_percent_count / metric_count,
        len(signed) / max(all_percent_count, 1),
        clipped.mean() if len(clipped) else 0.0,
        np.median(clipped) if len(clipped) else 0.0,
        np.quantile(clipped, 0.25) if len(clipped) else 0.0,
        np.quantile(clipped, 0.75) if len(clipped) else 0.0,
        np.mean(signed > 0.0) if len(signed) else 0.0,
        np.mean(signed < 0.0) if len(signed) else 0.0,
        *class_fractions,
        _matching_fraction(bodies, REPORTED_PATTERN),
        _matching_fraction(bodies, ESTIMATE_PATTERN),
        _matching_fraction(bodies, GUIDANCE_PATTERN),
        _matching_fraction(bodies, BEAT_PATTERN),
        _matching_fraction(bodies, MISS_PATTERN),
        _matching_fraction(bodies, FUTURE_PATTERN),
        np.log1p(len(level_values)),
        np.median(log_levels) if len(log_levels) else 0.0,
        np.log1p(_level_mode(level_values, company)),
        np.quantile(log_levels, 0.75) if len(log_levels) else 0.0,
        np.quantile(log_levels, 0.90) if len(log_levels) else 0.0,
        (np.quantile(log_levels, 0.75) - np.quantile(log_levels, 0.25))
        if len(log_levels) else 0.0,
    ], dtype=np.float32)
    if result.shape != (len(NUMERIC_FEATURE_NAMES),):
        raise AssertionError("Numeric feature schema mismatch")
    return result


def percentage_signal_probabilities(features, smoothing=0.05):
    start = NUMERIC_FEATURE_NAMES.index("percent_class_0_fraction")
    probabilities = np.asarray(features[start:start + 4], dtype=np.float64)
    probabilities += float(smoothing)
    return probabilities / probabilities.sum()
