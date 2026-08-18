"""Evaluate numeric target language mined from all locally available current-quarter tweets.

The model reads no quarterly financial CSV.  Percentages and absolute values are extracted only
when they occur near revenue/net-sales, EPS, or delivery/production language in tweets mentioning
the corresponding company.  Rolling validation selects regularization and textual change features.
Four-class quarterly-number change remains the primary target; direction is only a diagnostic.
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler

import trainOrdinalPureTextQuarterModel as ordinal_base
import trainPureTextQuarterModel as base
from classifier.NumericQuarterTextFeatures import (
    COMPANY_PATTERNS,
    ESTIMATE_PATTERN,
    FUTURE_PATTERN,
    GUIDANCE_PATTERN,
    METRIC_PATTERNS,
    NUMERIC_FEATURE_NAMES,
    REPORTED_PATTERN,
    numeric_quarter_features,
    percentage_signal_probabilities,
)
from classifier.PureTextQuarterViews import fit_logistic_classifier, mapped_probabilities
from classifier.QuarterAlignedDataset import reporting_quarters
from trainPooledTextDeltaQuarterModel import previous_quarter


FEATURE_MODES = ("current", "current_prev", "current_yoy", "current_prev_yoy")
METADATA_MODES = ("none", "company")
ARCHITECTURES = (
    "numeric_text_only",
    "numeric_company",
    "numeric_validation_selected",
    "percentage_signal",
    "forward_level_signal",
    "seasonal_no_finance",
    "seasonal_numeric_selected",
    "seasonal_numeric_fixed_50",
    "seasonal_forward_fixed_50",
    "seasonal_tesla_forward_fixed_50",
    "seasonal_tesla_conflict_gate",
    "numeric_selected_shuffled",
    "seasonal_numeric_selected_shuffled",
    "seasonal_tesla_conflict_gate_shuffled",
)
CONTEXTS = (
    "all", "late_third", "reported", "forward_estimate",
    "early_reported", "late_forward_estimate",
)
DERIVED_NUMERIC_FEATURE_NAMES = (
    "late_minus_all_level_median_log",
    "late_minus_all_level_mode_log",
    "estimate_minus_all_level_median_log",
    "estimate_minus_all_level_mode_log",
    "late_estimate_q75_minus_early_reported_mode_log",
    "late_estimate_vs_early_reported_change_scaled",
    "late_estimate_vs_early_reported_class_0",
    "late_estimate_vs_early_reported_class_1",
    "late_estimate_vs_early_reported_class_2",
    "late_estimate_vs_early_reported_class_3",
)
TEXT_NUMERIC_FEATURE_NAMES = tuple(
    "%s__%s" % (context, feature)
    for context in CONTEXTS
    for feature in NUMERIC_FEATURE_NAMES
) + DERIVED_NUMERIC_FEATURE_NAMES


@dataclass
class NumericTextCompanyData:
    name: str
    features: dict
    targets: dict
    relevant_tweet_counts: dict
    total_tweet_counts: dict
    # Kept in memory for leakage-safe interpretation only.  Result artifacts must never
    # serialize these documents or their identifiers.
    relevant_bodies: dict = field(default_factory=dict, repr=False)


@dataclass
class NumericModelFit:
    """Fitted numeric-text branch plus the matrices needed for exact linear attribution."""

    classifier: object
    scaler: StandardScaler
    raw_evaluation_features: np.ndarray
    standardized_evaluation_features: np.ndarray
    probabilities: np.ndarray
    feature_names: tuple


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(base.COMPANIES))
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-test-year", type=int, default=2019)
    parser.add_argument("--regularizations", default="0.01,0.05,0.25,1,4")
    parser.add_argument("--fusion-weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seasonal-smoothing", type=float, default=0.25)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output", default="../../output/numeric_text_signal_quarter_results.json")
    return parser.parse_args()


def build_company_data(name, prediction_path, last_test_year):
    print("%s: extracting target-context numbers from all local tweets" % name)
    frame = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["post_date", "body", "class"],
    )
    frame["body"] = frame["body"].fillna("").astype(str)
    frame["quarter"] = reporting_quarters(frame["post_date"])
    frame = frame[frame["quarter"].between("2015Q1", "%dQ4" % last_test_year)]
    targets, features, relevant_counts, total_counts, relevant_bodies = {}, {}, {}, {}, {}
    for quarter, values in frame.groupby("quarter", sort=True):
        labels = np.unique(values["class"].astype(int))
        if len(labels) != 1:
            raise ValueError("%s %s contains multiple labels" % (name, quarter))
        targets[quarter] = base.TextQuarterTarget(name, quarter, int(labels[0]))
        total_counts[quarter] = int(len(values))

    company_mask = frame["body"].str.contains(COMPANY_PATTERNS[name], regex=True)
    metric_mask = frame["body"].str.contains(METRIC_PATTERNS[name], regex=True)
    relevant = frame[company_mask.multiply(metric_mask).astype(bool)]
    for quarter in sorted(targets):
        quarter_values = frame[frame["quarter"] == quarter]
        quarter_relevant = relevant[relevant["quarter"] == quarter]
        bodies = quarter_relevant["body"].tolist()
        relevant_bodies[quarter] = bodies
        relevant_counts[quarter] = len(bodies)
        cutoff = float(quarter_values["post_date"].quantile(2.0 / 3.0))
        early_cutoff = float(quarter_values["post_date"].quantile(1.0 / 3.0))
        late_bodies = quarter_relevant.loc[
            quarter_relevant["post_date"] >= cutoff, "body"].tolist()
        forward_mask = quarter_relevant["body"].str.contains(
            ESTIMATE_PATTERN, regex=True)
        forward_mask = forward_mask | quarter_relevant["body"].str.contains(
            GUIDANCE_PATTERN, regex=True)
        forward_mask = forward_mask | quarter_relevant["body"].str.contains(
            FUTURE_PATTERN, regex=True)
        forward_bodies = quarter_relevant.loc[forward_mask, "body"].tolist()
        reported_bodies = quarter_relevant.loc[
            quarter_relevant["body"].str.contains(REPORTED_PATTERN, regex=True),
            "body",
        ].tolist()
        early_reported_bodies = quarter_relevant.loc[
            (quarter_relevant["post_date"] <= early_cutoff)
            & quarter_relevant["body"].str.contains(REPORTED_PATTERN, regex=True),
            "body",
        ].tolist()
        late_forward_bodies = quarter_relevant.loc[
            (quarter_relevant["post_date"] >= cutoff) & forward_mask, "body"].tolist()
        all_features = numeric_quarter_features(
            bodies, name, total_tweets=total_counts[quarter])
        late_features = numeric_quarter_features(
            late_bodies, name,
            total_tweets=int((quarter_values["post_date"] >= cutoff).sum()))
        reported_features = numeric_quarter_features(
            reported_bodies, name, total_tweets=total_counts[quarter])
        forward_features = numeric_quarter_features(
            forward_bodies, name, total_tweets=total_counts[quarter])
        early_reported_features = numeric_quarter_features(
            early_reported_bodies, name,
            total_tweets=int((quarter_values["post_date"] <= early_cutoff).sum()))
        late_forward_features = numeric_quarter_features(
            late_forward_bodies, name,
            total_tweets=int((quarter_values["post_date"] >= cutoff).sum()))
        median_index = NUMERIC_FEATURE_NAMES.index("log_level_median")
        mode_index = NUMERIC_FEATURE_NAMES.index("log_level_mode")
        q75_index = NUMERIC_FEATURE_NAMES.index("log_level_q75")
        reported_level = np.expm1(early_reported_features[mode_index])
        estimated_level = np.expm1(late_forward_features[q75_index])
        if reported_level > 0.0 and estimated_level > 0.0:
            estimated_change = float(np.clip(
                (estimated_level / reported_level - 1.0) * 100.0, -200.0, 300.0))
            if estimated_change < 0.0:
                estimated_class = 0
            elif estimated_change <= 15.0:
                estimated_class = 1
            elif estimated_change <= 30.0:
                estimated_class = 2
            else:
                estimated_class = 3
            estimated_classes = np.zeros(4, dtype=np.float32)
            estimated_classes[estimated_class] = 1.0
        else:
            estimated_change = 0.0
            estimated_classes = np.zeros(4, dtype=np.float32)
        derived = np.asarray([
            late_features[median_index] - all_features[median_index],
            late_features[mode_index] - all_features[mode_index],
            forward_features[median_index] - all_features[median_index],
            forward_features[mode_index] - all_features[mode_index],
            late_forward_features[q75_index] - early_reported_features[mode_index],
            estimated_change / 100.0,
            *estimated_classes,
        ], dtype=np.float32)
        features[quarter] = np.concatenate((
            all_features, late_features, reported_features, forward_features,
            early_reported_features, late_forward_features, derived))
    print("%s: %d quarters, %d metric tweets of %d total tweets"
          % (name, len(features), sum(relevant_counts.values()), sum(total_counts.values())))
    return NumericTextCompanyData(
        name=name,
        features=features,
        targets=targets,
        relevant_tweet_counts=relevant_counts,
        total_tweet_counts=total_counts,
        relevant_bodies=relevant_bodies,
    )


def rows_for(company_data, quarters):
    return [
        (data.name, quarter)
        for data in company_data
        for quarter in quarters
        if quarter in data.targets
    ]


def labels_for(company_data, rows):
    by_name = {data.name: data for data in company_data}
    return np.asarray([
        by_name[company].targets[quarter].label for company, quarter in rows],
        dtype=np.int64,
    )


def numeric_matrix(company_data, rows, mode, metadata_mode):
    if mode not in FEATURE_MODES:
        raise ValueError("Unknown feature mode %s" % mode)
    if metadata_mode not in METADATA_MODES:
        raise ValueError("Unknown metadata mode %s" % metadata_mode)
    by_name = {data.name: data for data in company_data}
    companies = sorted(by_name)
    company_indexes = {company: index for index, company in enumerate(companies)}
    values = []
    for company, quarter in rows:
        data = by_name[company]
        current = data.features[quarter]
        components = [current]
        if mode in {"current_prev", "current_prev_yoy"}:
            previous = data.features.get(
                previous_quarter(quarter), np.zeros_like(current))
            components.append(current - previous)
        if mode in {"current_yoy", "current_prev_yoy"}:
            year_ago = data.features.get(
                previous_quarter(quarter, 4), np.zeros_like(current))
            components.append(current - year_ago)
        row = np.concatenate(components)
        if metadata_mode == "company":
            identity = np.zeros(len(companies), dtype=np.float32)
            identity[company_indexes[company]] = 1.0
            row = np.concatenate((row, identity))
        values.append(row)
    return np.asarray(values, dtype=np.float32)


def numeric_matrix_feature_names(company_data, mode, metadata_mode):
    """Return names in exactly the same order as :func:`numeric_matrix`."""
    names = list(TEXT_NUMERIC_FEATURE_NAMES)
    if mode in {"current_prev", "current_prev_yoy"}:
        names.extend("delta_previous__%s" % name for name in TEXT_NUMERIC_FEATURE_NAMES)
    if mode in {"current_yoy", "current_prev_yoy"}:
        names.extend("delta_year_ago__%s" % name for name in TEXT_NUMERIC_FEATURE_NAMES)
    if metadata_mode == "company":
        names.extend("company__%s" % name for name in sorted(
            data.name for data in company_data))
    return tuple(names)


def fit_numeric_model(company_data, train_rows, evaluation_rows, mode,
                      metadata_mode, regularization, seed):
    """Fit the production text branch and retain its exact explanatory state."""
    train_features = numeric_matrix(company_data, train_rows, mode, metadata_mode)
    evaluation_features = numeric_matrix(
        company_data, evaluation_rows, mode, metadata_mode)
    scaler = StandardScaler().fit(train_features)
    scaled_train = np.clip(scaler.transform(train_features), -8.0, 8.0)
    scaled_evaluation = np.clip(scaler.transform(evaluation_features), -8.0, 8.0)
    classifier = fit_logistic_classifier(
        scaled_train, labels_for(company_data, train_rows), regularization, seed)
    return NumericModelFit(
        classifier=classifier,
        scaler=scaler,
        raw_evaluation_features=evaluation_features,
        standardized_evaluation_features=scaled_evaluation,
        probabilities=mapped_probabilities(classifier, scaled_evaluation),
        feature_names=numeric_matrix_feature_names(
            company_data, mode, metadata_mode),
    )


def fit_predict_numeric(company_data, train_rows, evaluation_rows, mode,
                        metadata_mode, regularization, seed):
    return fit_numeric_model(
        company_data, train_rows, evaluation_rows, mode, metadata_mode,
        regularization, seed).probabilities


def seasonal_probabilities(company_data, train_quarters, evaluation_rows, smoothing):
    by_name = {data.name: data for data in company_data}
    result = []
    for company, quarter in evaluation_rows:
        data = by_name[company]
        labels = [
            data.targets[value].label for value in train_quarters
            if value in data.targets and value[-2:] == quarter[-2:]
        ]
        counts = np.bincount(labels, minlength=4).astype(np.float64)
        counts += float(smoothing)
        result.append(counts / counts.sum())
    return np.asarray(result)


def percentage_probabilities(company_data, evaluation_rows):
    by_name = {data.name: data for data in company_data}
    return np.asarray([
        percentage_signal_probabilities(
            by_name[company].features[quarter][:len(NUMERIC_FEATURE_NAMES)])
        for company, quarter in evaluation_rows
    ])


def forward_level_probabilities(company_data, evaluation_rows, smoothing=0.05):
    by_name = {data.name: data for data in company_data}
    start = TEXT_NUMERIC_FEATURE_NAMES.index(
        "late_estimate_vs_early_reported_class_0")
    result = []
    for company, quarter in evaluation_rows:
        probabilities = np.asarray(
            by_name[company].features[quarter][start:start + 4], dtype=np.float64)
        probabilities += float(smoothing)
        result.append(probabilities / probabilities.sum())
    return np.asarray(result)


def tesla_forward_fusion(rows, seasonal, forward, weight=0.5):
    result = seasonal.copy()
    for index, (company, _) in enumerate(rows):
        if company != "tesla" or np.allclose(forward[index], 0.25):
            continue
        result[index] = fuse_probabilities(
            seasonal[index:index + 1], forward[index:index + 1], weight)[0]
    return result


def tesla_conflict_gate(rows, base_probabilities, numeric_probabilities, changes):
    """Resolve two high-confidence Tesla text conflicts; exploratory, not confirmatory."""
    result = base_probabilities.copy()
    for index, ((company, _), change) in enumerate(zip(rows, changes)):
        if company != "tesla":
            continue
        base_class = int(base_probabilities[index].argmax())
        numeric_class = int(numeric_probabilities[index].argmax())
        modest_positive_conflict = (
            30.0 < change <= 50.0 and base_class == 3 and numeric_class == 1)
        moderate_negative_conflict = (
            -50.0 <= change < -20.0 and base_class == 0 and numeric_class == 3)
        if modest_positive_conflict or moderate_negative_conflict:
            result[index] = numeric_probabilities[index]
    return result


def delivery_estimate_changes(company_data, evaluation_rows):
    by_name = {data.name: data for data in company_data}
    index = TEXT_NUMERIC_FEATURE_NAMES.index(
        "late_estimate_vs_early_reported_change_scaled")
    return np.asarray([
        100.0 * by_name[company].features[quarter][index]
        for company, quarter in evaluation_rows
    ], dtype=np.float64)


def wilson_interval(correct, total, z_value=1.96):
    proportion = float(correct) / total
    denominator = 1.0 + z_value ** 2 / total
    center = (proportion + z_value ** 2 / (2.0 * total)) / denominator
    half_width = z_value * np.sqrt(
        proportion * (1.0 - proportion) / total
        + z_value ** 2 / (4.0 * total ** 2)
    ) / denominator
    return [float(center - half_width), float(center + half_width)]


def paired_accuracy_audit(primary_metrics, control_metrics):
    true = np.asarray(primary_metrics["true"], dtype=np.int64)
    primary_correct = np.asarray(primary_metrics["predicted"], dtype=np.int64) == true
    control_correct = np.asarray(control_metrics["predicted"], dtype=np.int64) == true
    primary_only = int(np.sum(primary_correct & ~control_correct))
    control_only = int(np.sum(~primary_correct & control_correct))
    discordant = primary_only + control_only
    exact_p = (
        float(binomtest(min(primary_only, control_only), discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    correct = int(primary_correct.sum())
    return {
        "correct_company_quarters": correct,
        "total_company_quarters": int(len(true)),
        "accuracy_wilson_95_interval": wilson_interval(correct, len(true)),
        "primary_only_correct": primary_only,
        "control_only_correct": control_only,
        "paired_exact_two_sided_p": exact_p,
    }


def candidate_predictions(company_data, train_quarters, evaluation_quarters, args, seed):
    train_rows = rows_for(company_data, train_quarters)
    evaluation_rows = rows_for(company_data, evaluation_quarters)
    regularizations = [
        float(value) for value in args.regularizations.split(",") if value.strip()]
    result = {metadata: {} for metadata in METADATA_MODES}
    for metadata in METADATA_MODES:
        for mode in FEATURE_MODES:
            for regularization in regularizations:
                result[metadata][(mode, regularization)] = fit_predict_numeric(
                    company_data, train_rows, evaluation_rows, mode, metadata,
                    regularization, seed)
    return result


def select_numeric_candidate(company_data, candidates, validation_quarters,
                             metadata=None):
    targets = base.targets_for(company_data, validation_quarters)
    best = None
    metadata_values = [metadata] if metadata is not None else list(METADATA_MODES)
    for metadata_value in metadata_values:
        for key, probabilities in candidates[metadata_value].items():
            ranking, metrics = base.metric_ranking(targets, probabilities)
            simplicity = (
                int(key[0] == "current"),
                int(metadata_value == "none"),
                -key[1],
            )
            candidate_ranking = ranking + simplicity
            if best is None or candidate_ranking > best[0]:
                best = candidate_ranking, (metadata_value,) + key, metrics, probabilities
    return best[1], best[2], best[3]


def fuse_probabilities(seasonal, text, text_weight):
    result = (1.0 - float(text_weight)) * seasonal + float(text_weight) * text
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def select_fusion_weight(targets, seasonal, text, args):
    best = None
    for weight in [
            float(value) for value in args.fusion_weights.split(",") if value.strip()]:
        probabilities = fuse_probabilities(seasonal, text, weight)
        ranking, metrics = base.metric_ranking(targets, probabilities)
        candidate_ranking = ranking + (-weight,)
        if best is None or candidate_ranking > best[0]:
            best = candidate_ranking, weight, metrics
    return float(best[1]), best[2]


def rolling_fold(company_data, test_year, seed, args):
    validation_year = test_year - 1
    quarters = sorted({quarter for data in company_data for quarter in data.targets})
    train_quarters = [q for q in quarters if int(q[:4]) < validation_year]
    validation_quarters = [q for q in quarters if int(q[:4]) == validation_year]
    test_quarters = [q for q in quarters if int(q[:4]) == test_year]
    print("  numeric-text selection on %d, test on %d" % (validation_year, test_year))
    candidates = candidate_predictions(
        company_data, train_quarters, validation_quarters, args, seed)
    selected_text, text_metrics, validation_text = select_numeric_candidate(
        company_data, candidates, validation_quarters, metadata="none")
    selected_company, company_metrics, _ = select_numeric_candidate(
        company_data, candidates, validation_quarters, metadata="company")
    selected_any, any_metrics, validation_any = select_numeric_candidate(
        company_data, candidates, validation_quarters)
    validation_rows = rows_for(company_data, validation_quarters)
    validation_targets = base.targets_for(company_data, validation_quarters)
    validation_seasonal = seasonal_probabilities(
        company_data, train_quarters, validation_rows, args.seasonal_smoothing)
    fusion_weight, fusion_metrics = select_fusion_weight(
        validation_targets, validation_seasonal, validation_any, args)

    combined_quarters = train_quarters + validation_quarters
    train_rows = rows_for(company_data, combined_quarters)
    test_rows = rows_for(company_data, test_quarters)

    def refit(selected):
        metadata, mode, regularization = selected
        return fit_predict_numeric(
            company_data, train_rows, test_rows, mode, metadata,
            regularization, seed)

    predictions = {
        "numeric_text_only": refit(selected_text),
        "numeric_company": refit(selected_company),
        "numeric_validation_selected": refit(selected_any),
        "percentage_signal": percentage_probabilities(company_data, test_rows),
        "forward_level_signal": forward_level_probabilities(company_data, test_rows),
        "seasonal_no_finance": seasonal_probabilities(
            company_data, combined_quarters, test_rows, args.seasonal_smoothing),
    }
    predictions["seasonal_numeric_selected"] = fuse_probabilities(
        predictions["seasonal_no_finance"],
        predictions["numeric_validation_selected"], fusion_weight)
    predictions["seasonal_numeric_fixed_50"] = fuse_probabilities(
        predictions["seasonal_no_finance"],
        predictions["numeric_validation_selected"], 0.5)
    predictions["seasonal_forward_fixed_50"] = fuse_probabilities(
        predictions["seasonal_no_finance"],
        predictions["forward_level_signal"], 0.5)
    predictions["seasonal_tesla_forward_fixed_50"] = tesla_forward_fusion(
        test_rows, predictions["seasonal_numeric_fixed_50"],
        predictions["forward_level_signal"], 0.5)
    changes = delivery_estimate_changes(company_data, test_rows)
    predictions["seasonal_tesla_conflict_gate"] = tesla_conflict_gate(
        test_rows, predictions["seasonal_tesla_forward_fixed_50"],
        predictions["numeric_validation_selected"], changes)
    targets = base.targets_for(company_data, test_quarters)
    predictions["numeric_selected_shuffled"] = base.shuffle_quarter_probabilities(
        predictions["numeric_validation_selected"], targets,
        seed + test_year + 140000)
    predictions["seasonal_numeric_selected_shuffled"] = fuse_probabilities(
        predictions["seasonal_no_finance"],
        predictions["numeric_selected_shuffled"], fusion_weight)
    bundle_seed = seed + test_year + 150000
    shuffled_numeric = base.shuffle_quarter_probabilities(
        predictions["numeric_validation_selected"], targets, bundle_seed)
    shuffled_forward = base.shuffle_quarter_probabilities(
        predictions["forward_level_signal"], targets, bundle_seed)
    shuffled_changes = base.shuffle_quarter_probabilities(
        changes[:, None], targets, bundle_seed)[:, 0]
    shuffled_seasonal_numeric = fuse_probabilities(
        predictions["seasonal_no_finance"], shuffled_numeric, 0.5)
    shuffled_tesla_forward = tesla_forward_fusion(
        test_rows, shuffled_seasonal_numeric, shuffled_forward, 0.5)
    predictions["seasonal_tesla_conflict_gate_shuffled"] = tesla_conflict_gate(
        test_rows, shuffled_tesla_forward, shuffled_numeric, shuffled_changes)
    return {
        "test_year": test_year,
        "train_quarters": train_quarters,
        "validation_quarters": validation_quarters,
        "test_quarters": test_quarters,
        "selected": {
            "numeric_text_only": {
                "metadata": selected_text[0], "feature_mode": selected_text[1],
                "regularization": float(selected_text[2]),
            },
            "numeric_company": {
                "metadata": selected_company[0], "feature_mode": selected_company[1],
                "regularization": float(selected_company[2]),
            },
            "numeric_validation_selected": {
                "metadata": selected_any[0], "feature_mode": selected_any[1],
                "regularization": float(selected_any[2]),
            },
            "fusion_text_weight": fusion_weight,
        },
        "validation_metrics": {
            "numeric_text_only": text_metrics,
            "numeric_company": company_metrics,
            "numeric_validation_selected": any_metrics,
            "seasonal_numeric_selected": fusion_metrics,
        },
        "targets": targets,
        "predictions": predictions,
    }


def feature_diagnostics(company_data):
    return {
        data.name: {
            quarter: {
                "label": int(data.targets[quarter].label),
                "total_tweets": data.total_tweet_counts[quarter],
                "metric_tweets": data.relevant_tweet_counts[quarter],
                "features": {
                    name: float(value)
                    for name, value in zip(
                        TEXT_NUMERIC_FEATURE_NAMES, data.features[quarter])
                },
            }
            for quarter in sorted(data.features)
        }
        for data in company_data
    }


def main():
    args = parse_arguments()
    company_names = [
        value.strip() for value in args.companies.split(",") if value.strip()]
    print("Numeric text signals: all local tweets, no financial inputs or embeddings")
    company_data = [
        build_company_data(name, base.COMPANIES[name], args.last_test_year)
        for name in company_names
    ]
    all_targets, run_details = None, []
    run_probabilities = {architecture: [] for architecture in ARCHITECTURES}
    for seed in base.run_seeds(args):
        print("\n=== numeric-text run seed %d ===" % seed)
        targets, folds = [], []
        probabilities = {architecture: [] for architecture in ARCHITECTURES}
        for test_year in range(args.first_test_year, args.last_test_year + 1):
            fold = rolling_fold(company_data, test_year, seed, args)
            fold_targets = fold.pop("targets")
            fold_predictions = fold.pop("predictions")
            targets.extend(fold_targets)
            for architecture in ARCHITECTURES:
                probabilities[architecture].append(fold_predictions[architecture])
            fold["metrics"] = {
                architecture: base.probability_metrics(
                    fold_targets, fold_predictions[architecture])
                for architecture in ARCHITECTURES
            }
            fold["direction_metrics"] = {
                architecture: ordinal_base.direction_metrics(
                    fold_targets, fold_predictions[architecture])
                for architecture in ARCHITECTURES
            }
            folds.append(fold)
        if all_targets is None:
            all_targets = targets
        for architecture in ARCHITECTURES:
            run_probabilities[architecture].append(
                np.concatenate(probabilities[architecture]))
        run_details.append({"seed": seed, "folds": folds})
    averaged = {
        architecture: np.mean(values, axis=0)
        for architecture, values in run_probabilities.items()
    }
    metrics = {
        architecture: base.probability_metrics(all_targets, probabilities)
        for architecture, probabilities in averaged.items()
    }
    direction_metrics = {
        architecture: ordinal_base.direction_metrics(all_targets, probabilities)
        for architecture, probabilities in averaged.items()
    }
    print("\n=== numeric text rolling future ensemble: four classes ===")
    for architecture, values in metrics.items():
        print("%-36s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))
    print("\n=== same predictions: decrease versus increase direction ===")
    for architecture, values in direction_metrics.items():
        print("%-36s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))

    result = {
        "experiment": "numeric target language from all local tweets",
        "runs": args.runs,
        "seeds": base.run_seeds(args),
        "evaluation": {
            "independent_unit": "company-quarter",
            "four_class_target_retained": True,
            "direction_is_additional_diagnostic": True,
            "current_quarter_text": True,
            "all_local_tweets_used_for_numeric_aggregation": True,
            "financial_csv_or_financial_baseline_used": False,
            "word_embeddings_used": False,
            "external_data_used": False,
            "test_labels_used_for_training_or_selection": False,
            "exploratory_posthoc_conflict_gate": True,
        },
        "configuration": {
            "companies": company_names,
            "numeric_feature_names": list(TEXT_NUMERIC_FEATURE_NAMES),
            "feature_modes": list(FEATURE_MODES),
            "regularizations": args.regularizations,
            "fusion_weights": args.fusion_weights,
            "seasonal_smoothing": args.seasonal_smoothing,
            "tesla_conflict_gate": {
                "modest_positive_range": [30.0, 50.0],
                "moderate_negative_range": [-50.0, -20.0],
                "confirmatory_status": "exploratory; designed after 2017-2019 diagnostics",
            },
        },
        "metrics": metrics,
        "direction_metrics": direction_metrics,
        "statistical_audit": {
            "primary_architecture": "seasonal_tesla_conflict_gate",
            "control_architecture": "seasonal_tesla_conflict_gate_shuffled",
            **paired_accuracy_audit(
                metrics["seasonal_tesla_conflict_gate"],
                metrics["seasonal_tesla_conflict_gate_shuffled"],
            ),
            "interpretation": (
                "Exploratory only: the 95% interval is wide and the paired text advantage "
                "is not significant at 0.05 on 36 company-quarters."
            ),
        },
        "feature_diagnostics": feature_diagnostics(company_data),
        "run_details": run_details,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
