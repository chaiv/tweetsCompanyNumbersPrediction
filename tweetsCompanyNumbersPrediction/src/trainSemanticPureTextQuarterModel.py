"""Anti-shortcut, pure-text extension of the original tweet-group experiment.

URLs, domains, dates, usernames, cashtags and tracking boilerplate are removed before semantic
views are learned.  A finance-event view keeps only tweets containing target-independent business
language.  The important-word view scores words from quarter-level prevalence and requires them to
recur in at least two past quarters of a class.  No financial input or embedding is used.
"""

import argparse
import json
import os

import numpy as np

import trainPureTextQuarterModel as base
from classifier.PureTextQuarterViews import (
    QuarterStableImportantWords,
    aggregate_quarter_probabilities,
    fit_logistic_classifier,
    mapped_probabilities,
    semantic_word_vectorizer,
    word_vectorizer,
)


VIEWS = ("raw_word", "semantic_word", "finance_event_word", "stable_important_words")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(base.COMPANIES))
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--stable-max-features", type=int, default=40000)
    parser.add_argument("--regularizations", default="0.25,1,4")
    parser.add_argument("--temperatures", default="0.5,1,2")
    parser.add_argument("--important-word-count", type=int, default=25)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output", default="../../output/semantic_pure_text_quarter_results.json")
    return parser.parse_args()


def fit_candidate_views(data, seed, train_quarters, evaluation_quarters, args):
    train = base.record_arrays(base.selected_records(data, seed, train_quarters))
    evaluation = base.record_arrays(base.selected_records(data, seed, evaluation_quarters))
    regularizations = [
        float(value) for value in args.regularizations.split(",") if value.strip()]
    temperatures = [
        float(value) for value in args.temperatures.split(",") if value.strip()]
    candidates = {view: {} for view in VIEWS}
    text_views = (
        ("raw_word", "texts", word_vectorizer(args.max_features)),
        ("semantic_word", "semantic_texts", semantic_word_vectorizer(args.max_features)),
        ("finance_event_word", "finance_texts", semantic_word_vectorizer(args.max_features)),
    )
    for view, text_key, vectorizer in text_views:
        train_features = vectorizer.fit_transform(train[text_key])
        evaluation_features = vectorizer.transform(evaluation[text_key])
        for regularization in regularizations:
            classifier = fit_logistic_classifier(
                train_features, train["labels"], regularization, seed)
            candidates[view][regularization] = aggregate_quarter_probabilities(
                evaluation["quarters"],
                mapped_probabilities(classifier, evaluation_features),
                evaluation_quarters,
            )

    lexicon = QuarterStableImportantWords(
        max_features=args.stable_max_features).fit(
            train["semantic_texts"], train["labels"], train["quarters"])
    for temperature in temperatures:
        candidates["stable_important_words"][temperature] = (
            aggregate_quarter_probabilities(
                evaluation["quarters"],
                lexicon.predict_proba(
                    evaluation["semantic_texts"], temperature=temperature),
                evaluation_quarters,
            )
        )
    return candidates


def fit_selected_views(data, seed, train_quarters, test_quarters, selected, args):
    train = base.record_arrays(base.selected_records(data, seed, train_quarters))
    test = base.record_arrays(base.selected_records(data, seed, test_quarters))
    result = {}
    for view, text_key, vectorizer in (
        ("raw_word", "texts", word_vectorizer(args.max_features)),
        ("semantic_word", "semantic_texts", semantic_word_vectorizer(args.max_features)),
        ("finance_event_word", "finance_texts", semantic_word_vectorizer(args.max_features)),
    ):
        train_features = vectorizer.fit_transform(train[text_key])
        test_features = vectorizer.transform(test[text_key])
        classifier = fit_logistic_classifier(
            train_features, train["labels"], selected[view], seed)
        result[view] = aggregate_quarter_probabilities(
            test["quarters"], mapped_probabilities(classifier, test_features), test_quarters)

    lexicon = QuarterStableImportantWords(
        max_features=args.stable_max_features).fit(
            train["semantic_texts"], train["labels"], train["quarters"])
    result["stable_important_words"] = aggregate_quarter_probabilities(
        test["quarters"],
        lexicon.predict_proba(
            test["semantic_texts"], temperature=selected["stable_important_words"]),
        test_quarters,
    )
    return result, lexicon.top_words(args.important_word_count)


def select_candidates(company_data, candidate_results, validation_quarters):
    targets = base.targets_for(company_data, validation_quarters)
    selected, diagnostics, selected_probabilities = {}, {}, {}
    for view in VIEWS:
        values = sorted(next(iter(candidate_results.values()))[view])
        best = None
        for value in values:
            probabilities = np.concatenate([
                candidate_results[data.name][view][value] for data in company_data])
            ranking, metrics = base.metric_ranking(targets, probabilities)
            if best is None or ranking > best[0]:
                best = ranking, value, metrics, probabilities
        selected[view] = float(best[1])
        diagnostics[view] = best[2]
        selected_probabilities[view] = best[3]
    return selected, diagnostics, selected_probabilities


def select_single_view(validation_targets, probabilities):
    best = None
    for view in VIEWS:
        ranking, metrics = base.metric_ranking(validation_targets, probabilities[view])
        if best is None or ranking > best[0]:
            best = ranking, view, metrics
    return best[1], best[2]


def equal_ensemble(probabilities):
    result = np.mean([probabilities[view] for view in VIEWS], axis=0)
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


def rolling_fold(company_data, test_year, run_seed, args):
    validation_year = test_year - 1
    all_quarters = sorted({quarter for data in company_data for quarter in data.targets})
    train_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) < validation_year]
    validation_quarters = [
        quarter for quarter in all_quarters if int(quarter[:4]) == validation_year]
    test_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) == test_year]
    print("  semantic text selection on %d, test on %d" % (validation_year, test_year))

    candidates = {
        data.name: fit_candidate_views(
            data, run_seed, train_quarters, validation_quarters, args)
        for data in company_data
    }
    selected, validation_metrics, validation_probabilities = select_candidates(
        company_data, candidates, validation_quarters)
    validation_targets = base.targets_for(company_data, validation_quarters)
    selected_view, selected_view_metrics = select_single_view(
        validation_targets, validation_probabilities)

    combined_quarters = train_quarters + validation_quarters
    company_results, stable_words = {}, {}
    for data in company_data:
        company_results[data.name], stable_words[data.name] = fit_selected_views(
            data, run_seed, combined_quarters, test_quarters, selected, args)
    test_targets = base.targets_for(company_data, test_quarters)
    predictions = {
        view: np.concatenate([company_results[data.name][view] for data in company_data])
        for view in VIEWS
    }
    predictions["validation_selected_view"] = predictions[selected_view]
    predictions["equal_ensemble"] = equal_ensemble(predictions)
    predictions["selected_view_shuffled"] = base.shuffle_quarter_probabilities(
        predictions["validation_selected_view"], test_targets, run_seed + test_year + 70000)
    predictions["past_majority"] = base.past_majority_probabilities(
        company_data, combined_quarters, test_quarters)
    return {
        "test_year": test_year,
        "train_quarters": train_quarters,
        "validation_quarters": validation_quarters,
        "test_quarters": test_quarters,
        "selected_candidates": selected,
        "validation_selected_view": selected_view,
        "validation_metrics": {
            **validation_metrics,
            "validation_selected_view": selected_view_metrics,
        },
        "stable_important_words": stable_words,
        "targets": test_targets,
        "predictions": predictions,
    }


def main():
    args = parse_arguments()
    company_names = [value.strip() for value in args.companies.split(",") if value.strip()]
    for name in company_names:
        if name not in base.COMPANIES:
            raise ValueError("Unknown company %s" % name)
    print("Semantic pure text: anti-shortcut normalization, no finance or embeddings")
    args.include_style = False
    company_data = [
        base.build_company_data(name, index, base.COMPANIES[name], args)
        for index, name in enumerate(company_names)
    ]
    architectures = VIEWS + (
        "validation_selected_view", "equal_ensemble", "selected_view_shuffled",
        "past_majority")
    all_targets, run_details = None, []
    run_probabilities = {architecture: [] for architecture in architectures}
    for run_seed in base.run_seeds(args):
        print("\n=== semantic pure-text run seed %d ===" % run_seed)
        targets, folds = [], []
        probabilities = {architecture: [] for architecture in architectures}
        for test_year in range(args.first_test_year, args.last_test_year + 1):
            fold = rolling_fold(company_data, test_year, run_seed, args)
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
        elif [(target.company, target.quarter) for target in targets] != [
                (target.company, target.quarter) for target in all_targets]:
            raise AssertionError("Runs produced a different test-quarter order")
        for architecture in architectures:
            run_probabilities[architecture].append(np.concatenate(probabilities[architecture]))
        run_details.append({"seed": run_seed, "folds": folds})

    averaged = {
        architecture: np.mean(values, axis=0)
        for architecture, values in run_probabilities.items()
    }
    metrics = {
        architecture: base.probability_metrics(all_targets, probabilities)
        for architecture, probabilities in averaged.items()
    }
    print("\n=== semantic pure-text rolling future ensemble ===")
    for architecture, values in metrics.items():
        print("%-26s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))

    result = {
        "experiment": "anti-shortcut semantic pure-text quarter groups",
        "runs": args.runs,
        "seeds": base.run_seeds(args),
        "evaluation": {
            "first_test_year": args.first_test_year,
            "last_test_year": args.last_test_year,
            "independent_unit": "company-quarter",
            "current_quarter_text": True,
            "financial_inputs_or_baseline_used": False,
            "word_embeddings_used": False,
            "test_labels_used_for_training_or_selection": False,
        },
        "configuration": {
            "companies": company_names,
            "groups_per_quarter": args.groups_per_quarter,
            "regularizations": args.regularizations,
            "temperatures": args.temperatures,
            "stable_word_minimum_class_quarters": 2,
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
