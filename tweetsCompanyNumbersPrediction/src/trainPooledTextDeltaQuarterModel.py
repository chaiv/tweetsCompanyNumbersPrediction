"""Pooled quarter-level pure-text models with textual quarter changes.

Unlike group-level models, this experiment creates one sparse text row per independent
company-quarter.  A shared classifier can learn recurring language across Amazon, Apple and
Tesla.  Current text is optionally augmented with its sparse change from the preceding quarter
and the same quarter one year earlier.  Company identity is the only metadata.  No financial
input, financial baseline, embedding, external data, or future label is used.
"""

import argparse
import json
import os

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import trainOrdinalPureTextQuarterModel as ordinal_base
import trainPureTextQuarterModel as base
from classifier.PureTextQuarterViews import (
    SparseOrdinalClassifier,
    fit_logistic_classifier,
    mapped_probabilities,
    semantic_word_vectorizer,
    temperature_scale_probabilities,
)


TEXT_VIEWS = {
    "semantic_word": "semantic_texts",
    "finance_event_word": "finance_texts",
}
FEATURE_MODES = (
    "current",
    "current_prev",
    "current_yoy",
    "current_prev_yoy",
)
MODEL_FAMILIES = ("ordinal", "multinomial")
SELECTED_MODELS = {
    "pooled_ordinal_selected": ("ordinal", None),
    "pooled_multinomial_selected": ("multinomial", None),
    "pooled_multinomial_current": ("multinomial", "current"),
    "pooled_multinomial_text_delta": ("multinomial", "current_prev_yoy"),
    "pooled_ordinal_text_delta": ("ordinal", "current_prev_yoy"),
}
ARCHITECTURES = tuple(SELECTED_MODELS) + (
    "pooled_ordinal_selected_shuffled",
    "pooled_multinomial_selected_shuffled",
    "past_majority",
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(base.COMPANIES))
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--regularizations", default="0.05,0.25,1,4")
    parser.add_argument("--temperatures", default="1")
    parser.add_argument("--company-weights", default="0,1,3")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output", default="../../output/pooled_text_delta_quarter_results.json")
    return parser.parse_args()


def previous_quarter(quarter, lag=1):
    year, quarter_index = int(quarter[:4]), int(quarter[-1]) - 1
    absolute_index = year * 4 + quarter_index - int(lag)
    return "%dQ%d" % (absolute_index // 4, absolute_index % 4 + 1)


def target_rows(company_data, quarters):
    return [
        (data.name, quarter)
        for data in company_data
        for quarter in quarters
        if quarter in data.targets
    ]


def pooled_training_texts(company_data, seed, quarters, text_key):
    texts = []
    for data in company_data:
        arrays = base.record_arrays(base.selected_records(data, seed, quarters))
        texts.extend(arrays[text_key])
    return texts


def quarter_feature_maps(company_data, seed, quarters, text_key, vectorizer):
    """Return a sparse mean TF-IDF row for each selected company-quarter."""
    result = {}
    for data in company_data:
        arrays = base.record_arrays(base.selected_records(data, seed, quarters))
        features = vectorizer.transform(arrays[text_key])
        group_quarters = np.asarray(arrays["quarters"])
        for quarter in quarters:
            indexes = np.flatnonzero(group_quarters == quarter)
            if not len(indexes):
                continue
            result[(data.name, quarter)] = csr_matrix(
                features[indexes].mean(axis=0), dtype=np.float32)
    return result


def quarter_feature_matrix(feature_maps, rows, companies, mode, company_weight):
    if mode not in FEATURE_MODES:
        raise ValueError("Unknown feature mode %s" % mode)
    feature_count = next(iter(feature_maps.values())).shape[1]
    zero = csr_matrix((1, feature_count), dtype=np.float32)
    values = []
    for company, quarter in rows:
        current = feature_maps[(company, quarter)]
        components = [current]
        if mode in {"current_prev", "current_prev_yoy"}:
            previous = feature_maps.get((company, previous_quarter(quarter)), zero)
            components.append(current - previous)
        if mode in {"current_yoy", "current_prev_yoy"}:
            year_ago = feature_maps.get((company, previous_quarter(quarter, 4)), zero)
            components.append(current - year_ago)
        values.append(hstack(components, format="csr", dtype=np.float32))
    text_features = vstack(values, format="csr", dtype=np.float32)
    company_indexes = {company: index for index, company in enumerate(companies)}
    row_indexes = np.arange(len(rows))
    column_indexes = np.asarray([company_indexes[company] for company, _ in rows])
    company_features = csr_matrix(
        (np.full(len(rows), float(company_weight), dtype=np.float32),
         (row_indexes, column_indexes)),
        shape=(len(rows), len(companies)),
    )
    return hstack((text_features, company_features), format="csr", dtype=np.float32)


def labels_for_rows(company_data, rows):
    by_name = {data.name: data for data in company_data}
    return np.asarray([
        by_name[company].targets[quarter].label for company, quarter in rows],
        dtype=np.int64,
    )


def fit_predict(family, train_features, train_labels, evaluation_features,
                regularization, temperature, seed):
    if family == "ordinal":
        classifier = SparseOrdinalClassifier(
            regularization=regularization, seed=seed).fit(
                train_features, train_labels)
        probabilities = classifier.predict_proba(evaluation_features)
    else:
        classifier = fit_logistic_classifier(
            train_features, train_labels, regularization, seed)
        probabilities = mapped_probabilities(classifier, evaluation_features)
    return temperature_scale_probabilities(probabilities, temperature)


def parsed_grid(args):
    regularizations = [
        float(value) for value in args.regularizations.split(",") if value.strip()]
    temperatures = [
        float(value) for value in args.temperatures.split(",") if value.strip()]
    company_weights = [
        float(value) for value in args.company_weights.split(",") if value.strip()]
    return regularizations, temperatures, company_weights


def candidate_predictions(company_data, seed, train_quarters,
                          evaluation_quarters, args):
    companies = [data.name for data in company_data]
    train_rows = target_rows(company_data, train_quarters)
    evaluation_rows = target_rows(company_data, evaluation_quarters)
    train_labels = labels_for_rows(company_data, train_rows)
    available_quarters = sorted({
        quarter for data in company_data for quarter in data.targets
        if quarter <= max(evaluation_quarters)
    })
    regularizations, temperatures, company_weights = parsed_grid(args)
    result = {family: {} for family in MODEL_FAMILIES}
    for view, text_key in TEXT_VIEWS.items():
        vectorizer = semantic_word_vectorizer(args.max_features)
        vectorizer.fit(pooled_training_texts(
            company_data, seed, train_quarters, text_key))
        maps = quarter_feature_maps(
            company_data, seed, available_quarters, text_key, vectorizer)
        for mode in FEATURE_MODES:
            for company_weight in company_weights:
                train_features = quarter_feature_matrix(
                    maps, train_rows, companies, mode, company_weight)
                evaluation_features = quarter_feature_matrix(
                    maps, evaluation_rows, companies, mode, company_weight)
                for regularization in regularizations:
                    for family in MODEL_FAMILIES:
                        for temperature in temperatures:
                            key = (
                                view, mode, regularization, temperature, company_weight)
                            result[family][key] = fit_predict(
                                family, train_features, train_labels,
                                evaluation_features, regularization, temperature, seed)
    return result


def select_candidate(company_data, candidate_results, validation_quarters,
                     family, required_mode=None):
    targets = base.targets_for(company_data, validation_quarters)
    best = None
    keys = sorted(next(iter(candidate_results.values()))[family])
    for key in keys:
        if required_mode is not None and key[1] != required_mode:
            continue
        probabilities = candidate_results["pooled"][family][key]
        ranking, metrics = base.metric_ranking(targets, probabilities)
        simplicity = (
            int(key[0] == "finance_event_word"),
            int(key[1] == "current"),
            -abs(np.log2(key[3])),
            -key[4],
        )
        candidate_ranking = ranking + simplicity
        if best is None or candidate_ranking > best[0]:
            best = candidate_ranking, key, metrics
    return best[1], best[2]


def refit_predictions(company_data, seed, train_quarters, test_quarters,
                      selected, args):
    companies = [data.name for data in company_data]
    train_rows = target_rows(company_data, train_quarters)
    test_rows = target_rows(company_data, test_quarters)
    train_labels = labels_for_rows(company_data, train_rows)
    available_quarters = sorted({
        quarter for data in company_data for quarter in data.targets
        if quarter <= max(test_quarters)
    })
    result = {}
    for view in sorted({key[0] for key in selected.values()}):
        text_key = TEXT_VIEWS[view]
        vectorizer = semantic_word_vectorizer(args.max_features)
        vectorizer.fit(pooled_training_texts(
            company_data, seed, train_quarters, text_key))
        maps = quarter_feature_maps(
            company_data, seed, available_quarters, text_key, vectorizer)
        for name, key in selected.items():
            if key[0] != view:
                continue
            _, mode, regularization, temperature, company_weight = key
            train_features = quarter_feature_matrix(
                maps, train_rows, companies, mode, company_weight)
            test_features = quarter_feature_matrix(
                maps, test_rows, companies, mode, company_weight)
            family = SELECTED_MODELS[name][0]
            result[name] = fit_predict(
                family, train_features, train_labels, test_features,
                regularization, temperature, seed)
    return result


def rolling_fold(company_data, test_year, seed, args):
    validation_year = test_year - 1
    quarters = sorted({quarter for data in company_data for quarter in data.targets})
    train_quarters = [
        quarter for quarter in quarters if int(quarter[:4]) < validation_year]
    validation_quarters = [
        quarter for quarter in quarters if int(quarter[:4]) == validation_year]
    test_quarters = [
        quarter for quarter in quarters if int(quarter[:4]) == test_year]
    print("  pooled text-delta selection on %d, test on %d"
          % (validation_year, test_year))
    candidates = {"pooled": candidate_predictions(
        company_data, seed, train_quarters, validation_quarters, args)}
    selected, validation_metrics = {}, {}
    for name, (family, mode) in SELECTED_MODELS.items():
        selected[name], validation_metrics[name] = select_candidate(
            company_data, candidates, validation_quarters, family, mode)

    combined_quarters = train_quarters + validation_quarters
    predictions = refit_predictions(
        company_data, seed, combined_quarters, test_quarters, selected, args)
    targets = base.targets_for(company_data, test_quarters)
    predictions["pooled_ordinal_selected_shuffled"] = (
        base.shuffle_quarter_probabilities(
            predictions["pooled_ordinal_selected"], targets,
            seed + test_year + 120000)
    )
    predictions["pooled_multinomial_selected_shuffled"] = (
        base.shuffle_quarter_probabilities(
            predictions["pooled_multinomial_selected"], targets,
            seed + test_year + 130000)
    )
    predictions["past_majority"] = base.past_majority_probabilities(
        company_data, combined_quarters, test_quarters)
    return {
        "test_year": test_year,
        "train_quarters": train_quarters,
        "validation_quarters": validation_quarters,
        "test_quarters": test_quarters,
        "selected": {
            name: {
                "view": key[0],
                "feature_mode": key[1],
                "regularization": float(key[2]),
                "temperature": float(key[3]),
                "company_weight": float(key[4]),
            }
            for name, key in selected.items()
        },
        "validation_metrics": validation_metrics,
        "targets": targets,
        "predictions": predictions,
    }


def main():
    args = parse_arguments()
    company_names = [
        value.strip() for value in args.companies.split(",") if value.strip()]
    for name in company_names:
        if name not in base.COMPANIES:
            raise ValueError("Unknown company %s" % name)
    args.include_style = False
    print("Pooled quarter text deltas: one row per company-quarter, no finance")
    company_data = [
        base.build_company_data(name, index, base.COMPANIES[name], args)
        for index, name in enumerate(company_names)
    ]
    all_targets, run_details = None, []
    run_probabilities = {architecture: [] for architecture in ARCHITECTURES}
    for seed in base.run_seeds(args):
        print("\n=== pooled quarter-text run seed %d ===" % seed)
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
        elif [(target.company, target.quarter) for target in targets] != [
                (target.company, target.quarter) for target in all_targets]:
            raise AssertionError("Runs produced a different test-quarter order")
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
    print("\n=== pooled text-delta rolling future ensemble: four classes ===")
    for architecture, values in metrics.items():
        print("%-38s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))
    print("\n=== same predictions: decrease versus increase direction ===")
    for architecture, values in direction_metrics.items():
        print("%-38s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))

    result = {
        "experiment": "pooled quarter-level pure-text deltas",
        "runs": args.runs,
        "seeds": base.run_seeds(args),
        "evaluation": {
            "first_test_year": args.first_test_year,
            "last_test_year": args.last_test_year,
            "independent_unit": "company-quarter",
            "one_training_row_per_independent_company_quarter": True,
            "four_class_target_retained": True,
            "direction_is_additional_diagnostic": True,
            "current_quarter_text": True,
            "financial_inputs_or_baseline_used": False,
            "word_embeddings_used": False,
            "external_data_used": False,
            "test_labels_used_for_training_or_selection": False,
        },
        "configuration": {
            "companies": company_names,
            "groups_per_quarter_for_text_estimation": args.groups_per_quarter,
            "text_views": list(TEXT_VIEWS),
            "feature_modes": list(FEATURE_MODES),
            "regularizations": args.regularizations,
            "temperatures": args.temperatures,
            "company_weights": args.company_weights,
        },
        "metrics": metrics,
        "direction_metrics": direction_metrics,
        "run_details": run_details,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
