"""Pure finance-event text with validation-selected group-to-quarter aggregation."""

import argparse
import json
import os

import numpy as np

import trainPureTextQuarterModel as base
from classifier.PureTextQuarterViews import (
    TEMPORAL_AGGREGATION_MODES,
    aggregate_temporal_quarter_probabilities,
    fit_logistic_classifier,
    mapped_probabilities,
    semantic_word_vectorizer,
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
    parser.add_argument("--aggregation-modes", default=",".join(TEMPORAL_AGGREGATION_MODES))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output", default="../../output/temporal_aggregation_pure_text_results.json")
    return parser.parse_args()


def candidate_predictions(data, seed, train_quarters, evaluation_quarters, args):
    train = base.record_arrays(base.selected_records(data, seed, train_quarters))
    evaluation = base.record_arrays(base.selected_records(data, seed, evaluation_quarters))
    vectorizer = semantic_word_vectorizer(args.max_features)
    train_features = vectorizer.fit_transform(train["finance_texts"])
    evaluation_features = vectorizer.transform(evaluation["finance_texts"])
    regularizations = [
        float(value) for value in args.regularizations.split(",") if value.strip()]
    temperatures = [
        float(value) for value in args.temperatures.split(",") if value.strip()]
    modes = [value.strip() for value in args.aggregation_modes.split(",") if value.strip()]
    result = {}
    for regularization in regularizations:
        classifier = fit_logistic_classifier(
            train_features, train["labels"], regularization, seed)
        group_probabilities = mapped_probabilities(classifier, evaluation_features)
        for temperature in temperatures:
            for mode in modes:
                key = (regularization, temperature, mode)
                result[key] = aggregate_temporal_quarter_probabilities(
                    evaluation["quarters"], evaluation["timestamps"], group_probabilities,
                    evaluation_quarters, mode=mode, temperature=temperature)
    return result


def select_candidate(company_data, candidate_results, validation_quarters):
    targets = base.targets_for(company_data, validation_quarters)
    best = None
    for candidate in next(iter(candidate_results.values())):
        probabilities = np.concatenate([
            candidate_results[data.name][candidate] for data in company_data])
        ranking, metrics = base.metric_ranking(targets, probabilities)
        # Prefer the simplest mean aggregation when validation metrics tie exactly.
        simplicity = int(candidate[2] == "mean")
        candidate_ranking = ranking + (simplicity, -candidate[0])
        if best is None or candidate_ranking > best[0]:
            best = candidate_ranking, candidate, metrics
    return best[1], best[2]


def refit_company(data, seed, train_quarters, test_quarters, selected, args):
    train = base.record_arrays(base.selected_records(data, seed, train_quarters))
    test = base.record_arrays(base.selected_records(data, seed, test_quarters))
    vectorizer = semantic_word_vectorizer(args.max_features)
    train_features = vectorizer.fit_transform(train["finance_texts"])
    test_features = vectorizer.transform(test["finance_texts"])
    classifier = fit_logistic_classifier(
        train_features, train["labels"], selected[0], seed)
    group_probabilities = mapped_probabilities(classifier, test_features)

    def aggregate(mode, temperature):
        return aggregate_temporal_quarter_probabilities(
            test["quarters"], test["timestamps"], group_probabilities, test_quarters,
            mode=mode, temperature=temperature)

    return {
        "temporal_selected": aggregate(selected[2], selected[1]),
        "finance_mean": aggregate("mean", 1.0),
        "finance_vote": aggregate("vote", 1.0),
        "finance_late_half": aggregate("late_half_mean", 1.0),
    }


def rolling_fold(company_data, test_year, seed, args):
    validation_year = test_year - 1
    quarters = sorted({quarter for data in company_data for quarter in data.targets})
    train_quarters = [quarter for quarter in quarters if int(quarter[:4]) < validation_year]
    validation_quarters = [
        quarter for quarter in quarters if int(quarter[:4]) == validation_year]
    test_quarters = [quarter for quarter in quarters if int(quarter[:4]) == test_year]
    print("  temporal aggregation selection on %d, test on %d"
          % (validation_year, test_year))
    candidates = {
        data.name: candidate_predictions(
            data, seed, train_quarters, validation_quarters, args)
        for data in company_data
    }
    selected, validation_metrics = select_candidate(
        company_data, candidates, validation_quarters)
    combined_quarters = train_quarters + validation_quarters
    company_predictions = {
        data.name: refit_company(data, seed, combined_quarters, test_quarters, selected, args)
        for data in company_data
    }
    targets = base.targets_for(company_data, test_quarters)
    predictions = {
        architecture: np.concatenate([
            company_predictions[data.name][architecture] for data in company_data])
        for architecture in (
            "temporal_selected", "finance_mean", "finance_vote", "finance_late_half")
    }
    predictions["temporal_selected_shuffled"] = base.shuffle_quarter_probabilities(
        predictions["temporal_selected"], targets, seed + test_year + 90000)
    predictions["past_majority"] = base.past_majority_probabilities(
        company_data, combined_quarters, test_quarters)
    return {
        "test_year": test_year,
        "train_quarters": train_quarters,
        "validation_quarters": validation_quarters,
        "test_quarters": test_quarters,
        "selected": {
            "regularization": float(selected[0]),
            "temperature": float(selected[1]),
            "aggregation_mode": selected[2],
        },
        "validation_metrics": validation_metrics,
        "targets": targets,
        "predictions": predictions,
    }


def main():
    args = parse_arguments()
    company_names = [value.strip() for value in args.companies.split(",") if value.strip()]
    args.include_style = False
    print("Temporal pure text: finance-event text, no financial inputs or embeddings")
    company_data = [
        base.build_company_data(name, index, base.COMPANIES[name], args)
        for index, name in enumerate(company_names)
    ]
    architectures = (
        "temporal_selected", "finance_mean", "finance_vote", "finance_late_half",
        "temporal_selected_shuffled", "past_majority")
    all_targets, run_details = None, []
    run_probabilities = {architecture: [] for architecture in architectures}
    for seed in base.run_seeds(args):
        print("\n=== temporal pure-text run seed %d ===" % seed)
        targets, folds = [], []
        probabilities = {architecture: [] for architecture in architectures}
        for test_year in range(args.first_test_year, args.last_test_year + 1):
            fold = rolling_fold(company_data, test_year, seed, args)
            fold_targets = fold.pop("targets")
            fold_predictions = fold.pop("predictions")
            targets.extend(fold_targets)
            for architecture in architectures:
                probabilities[architecture].append(fold_predictions[architecture])
            fold["metrics"] = {
                architecture: base.probability_metrics(
                    fold_targets, fold_predictions[architecture])
                for architecture in architectures
            }
            folds.append(fold)
        if all_targets is None:
            all_targets = targets
        for architecture in architectures:
            run_probabilities[architecture].append(np.concatenate(probabilities[architecture]))
        run_details.append({"seed": seed, "folds": folds})
    averaged = {
        architecture: np.mean(values, axis=0)
        for architecture, values in run_probabilities.items()
    }
    metrics = {
        architecture: base.probability_metrics(all_targets, probabilities)
        for architecture, probabilities in averaged.items()
    }
    print("\n=== temporal aggregation rolling future ensemble ===")
    for architecture, values in metrics.items():
        print("%-28s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))
    result = {
        "experiment": "temporal aggregation of pure finance-event text groups",
        "runs": args.runs,
        "seeds": base.run_seeds(args),
        "evaluation": {
            "financial_inputs_or_baseline_used": False,
            "word_embeddings_used": False,
            "test_labels_used_for_training_or_selection": False,
            "independent_unit": "company-quarter",
        },
        "configuration": {
            "companies": company_names,
            "groups_per_quarter": args.groups_per_quarter,
            "regularizations": args.regularizations,
            "temperatures": args.temperatures,
            "aggregation_modes": args.aggregation_modes,
        },
        "metrics": metrics,
        "run_details": run_details,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
