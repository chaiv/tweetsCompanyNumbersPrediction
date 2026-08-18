"""Strict rolling evaluation of ordered, pure-text quarterly predictions.

The four target classes are kept unchanged, but are learned through three ordered sparse
decisions: P(class > 0), P(class > 1), and P(class > 2).  Raw, anti-shortcut semantic, and
finance-event tweet-group text are compared using only the preceding validation year.  No
financial values, financial baseline, embeddings, or external data are model inputs.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, matthews_corrcoef

import trainPureTextQuarterModel as base
from classifier.PureTextQuarterViews import (
    SparseOrdinalClassifier,
    aggregate_quarter_probabilities,
    fit_logistic_classifier,
    mapped_probabilities,
    semantic_word_vectorizer,
    temperature_scale_probabilities,
    word_vectorizer,
)


TEXT_VIEWS = {
    "raw_word": "texts",
    "semantic_word": "semantic_texts",
    "finance_event_word": "finance_texts",
}
MODEL_FAMILIES = ("ordinal", "multinomial")
ARCHITECTURES = (
    "ordinal_selected",
    "multinomial_selected",
    "ordinal_selected_shuffled",
    "multinomial_selected_shuffled",
    "past_majority",
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(base.COMPANIES))
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--regularizations", default="0.25,1,4")
    parser.add_argument("--temperatures", default="0.5,1,2")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output", default="../../output/ordinal_pure_text_quarter_results.json")
    return parser.parse_args()


def vectorizer_for(view, max_features):
    if view == "raw_word":
        return word_vectorizer(max_features)
    return semantic_word_vectorizer(max_features)


def parsed_grid(args):
    regularizations = [
        float(value) for value in args.regularizations.split(",") if value.strip()]
    temperatures = [
        float(value) for value in args.temperatures.split(",") if value.strip()]
    return regularizations, temperatures


def fitted_group_probabilities(family, features, labels, evaluation_features,
                               regularization, seed):
    if family == "ordinal":
        classifier = SparseOrdinalClassifier(
            regularization=regularization, seed=seed).fit(features, labels)
        return classifier.predict_proba(evaluation_features)
    classifier = fit_logistic_classifier(features, labels, regularization, seed)
    return mapped_probabilities(classifier, evaluation_features)


def candidate_predictions(data, seed, train_quarters, evaluation_quarters, args):
    train = base.record_arrays(base.selected_records(data, seed, train_quarters))
    evaluation = base.record_arrays(
        base.selected_records(data, seed, evaluation_quarters))
    regularizations, temperatures = parsed_grid(args)
    result = {family: {} for family in MODEL_FAMILIES}
    for view, text_key in TEXT_VIEWS.items():
        vectorizer = vectorizer_for(view, args.max_features)
        train_features = vectorizer.fit_transform(train[text_key])
        evaluation_features = vectorizer.transform(evaluation[text_key])
        for regularization in regularizations:
            for family in MODEL_FAMILIES:
                probabilities = fitted_group_probabilities(
                    family, train_features, train["labels"], evaluation_features,
                    regularization, seed)
                for temperature in temperatures:
                    key = (view, regularization, temperature)
                    result[family][key] = aggregate_quarter_probabilities(
                        evaluation["quarters"],
                        temperature_scale_probabilities(probabilities, temperature),
                        evaluation_quarters,
                    )
    return result


def select_candidates(company_data, candidate_results, validation_quarters):
    targets = base.targets_for(company_data, validation_quarters)
    selected, diagnostics = {}, {}
    for family in MODEL_FAMILIES:
        keys = sorted(next(iter(candidate_results.values()))[family])
        best = None
        for key in keys:
            probabilities = np.concatenate([
                candidate_results[data.name][family][key] for data in company_data])
            ranking, metrics = base.metric_ranking(targets, probabilities)
            # Resolve exact ties toward semantic normalization, unit temperature and stronger
            # regularization; this does not inspect the future test labels.
            simplicity = (
                int(key[0] != "raw_word"),
                -abs(np.log2(key[2])),
                key[1],
            )
            candidate_ranking = ranking + simplicity
            if best is None or candidate_ranking > best[0]:
                best = candidate_ranking, key, metrics
        selected[family] = best[1]
        diagnostics[family] = best[2]
    return selected, diagnostics


def refit_company(data, seed, train_quarters, test_quarters, selected, args):
    train = base.record_arrays(base.selected_records(data, seed, train_quarters))
    test = base.record_arrays(base.selected_records(data, seed, test_quarters))
    result = {}
    for family in MODEL_FAMILIES:
        view, regularization, temperature = selected[family]
        text_key = TEXT_VIEWS[view]
        vectorizer = vectorizer_for(view, args.max_features)
        train_features = vectorizer.fit_transform(train[text_key])
        test_features = vectorizer.transform(test[text_key])
        group_probabilities = fitted_group_probabilities(
            family, train_features, train["labels"], test_features,
            regularization, seed)
        result[family + "_selected"] = aggregate_quarter_probabilities(
            test["quarters"],
            temperature_scale_probabilities(group_probabilities, temperature),
            test_quarters,
        )
    return result


def direction_metrics(targets, four_class_probabilities):
    true = np.asarray([target.label > 0 for target in targets], dtype=np.int64)
    probabilities = np.column_stack((
        four_class_probabilities[:, 0],
        four_class_probabilities[:, 1:].sum(axis=1),
    ))
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    predicted = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(true, predicted)),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "log_loss": float(log_loss(true, probabilities, labels=np.arange(2))),
        "true": true.tolist(),
        "predicted": predicted.tolist(),
        "probabilities": probabilities.tolist(),
        "companies": [target.company for target in targets],
        "quarters": [target.quarter for target in targets],
        "per_company": {},
    }
    for company in sorted({target.company for target in targets}):
        indexes = [
            index for index, target in enumerate(targets) if target.company == company]
        result["per_company"][company] = {
            "accuracy": float(accuracy_score(true[indexes], predicted[indexes])),
            "mcc": float(matthews_corrcoef(true[indexes], predicted[indexes])),
        }
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
    print("  ordinal/text selection on %d, test on %d"
          % (validation_year, test_year))

    candidates = {
        data.name: candidate_predictions(
            data, seed, train_quarters, validation_quarters, args)
        for data in company_data
    }
    selected, validation_metrics = select_candidates(
        company_data, candidates, validation_quarters)
    combined_quarters = train_quarters + validation_quarters
    company_predictions = {
        data.name: refit_company(
            data, seed, combined_quarters, test_quarters, selected, args)
        for data in company_data
    }
    targets = base.targets_for(company_data, test_quarters)
    predictions = {
        family + "_selected": np.concatenate([
            company_predictions[data.name][family + "_selected"]
            for data in company_data])
        for family in MODEL_FAMILIES
    }
    for family in MODEL_FAMILIES:
        name = family + "_selected"
        predictions[name + "_shuffled"] = base.shuffle_quarter_probabilities(
            predictions[name], targets, seed + test_year + 110000)
    predictions["past_majority"] = base.past_majority_probabilities(
        company_data, combined_quarters, test_quarters)
    return {
        "test_year": test_year,
        "train_quarters": train_quarters,
        "validation_quarters": validation_quarters,
        "test_quarters": test_quarters,
        "selected": {
            family: {
                "view": selected[family][0],
                "regularization": float(selected[family][1]),
                "temperature": float(selected[family][2]),
            }
            for family in MODEL_FAMILIES
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
    print("Ordinal pure text: 4 classes plus direction diagnostic; no finance or embeddings")
    company_data = [
        base.build_company_data(name, index, base.COMPANIES[name], args)
        for index, name in enumerate(company_names)
    ]

    all_targets, run_details = None, []
    run_probabilities = {architecture: [] for architecture in ARCHITECTURES}
    for seed in base.run_seeds(args):
        print("\n=== ordinal pure-text run seed %d ===" % seed)
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
                architecture: direction_metrics(
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
    binary_metrics = {
        architecture: direction_metrics(all_targets, probabilities)
        for architecture, probabilities in averaged.items()
    }
    print("\n=== ordinal pure-text rolling future ensemble: four classes ===")
    for architecture, values in metrics.items():
        print("%-31s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))
    print("\n=== same predictions: decrease versus increase direction ===")
    for architecture, values in binary_metrics.items():
        print("%-31s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))

    result = {
        "experiment": "ordered pure-text quarterly target",
        "runs": args.runs,
        "seeds": base.run_seeds(args),
        "evaluation": {
            "first_test_year": args.first_test_year,
            "last_test_year": args.last_test_year,
            "independent_unit": "company-quarter",
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
            "groups_per_quarter": args.groups_per_quarter,
            "text_views": list(TEXT_VIEWS),
            "regularizations": args.regularizations,
            "temperatures": args.temperatures,
            "ordinal_thresholds": ["class > 0", "class > 1", "class > 2"],
        },
        "metrics": metrics,
        "direction_metrics": binary_metrics,
        "run_details": run_details,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
