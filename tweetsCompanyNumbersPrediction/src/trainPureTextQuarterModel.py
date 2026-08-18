"""Strictly future-evaluated, pure-text version of the original tweet-group idea.

No financial history, financial baseline, word embedding or neural representation is an input.
Past quarter-aligned tweet groups train four complementary views: word TF-IDF, character TF-IDF,
past-only important-word log odds, and linguistic/sentiment/author style.  Group probabilities are
averaged to one decision per company-quarter.  Hyperparameters and late-fusion weights use only the
preceding validation year before models are refit and evaluated on the next untouched year.
"""

import argparse
import itertools
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, matthews_corrcoef
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from PredictionModelPath import (
    AMAZON_REVENUE_10_LSTM_MULTI_CLASS,
    APPLE__EPS_10_LSTM_MULTI_CLASS,
    TESLA_CAR_SALES_10_LSTM_MULTI_CLASS,
)
from classifier.PureTextQuarterViews import (
    ImportantWordLogOdds,
    STYLE_FEATURE_NAMES,
    aggregate_quarter_probabilities,
    character_vectorizer,
    fit_logistic_classifier,
    fit_style_classifier,
    finance_relevant_semantic_text,
    group_style_features,
    mapped_probabilities,
    normalize_semantic_text,
    style_probabilities,
    word_vectorizer,
)
from classifier.QuarterAlignedDataset import (
    build_quarter_groups,
    select_balanced_quarter_groups,
)


COMPANIES = {
    "amazon": AMAZON_REVENUE_10_LSTM_MULTI_CLASS,
    "apple": APPLE__EPS_10_LSTM_MULTI_CLASS,
    "tesla": TESLA_CAR_SALES_10_LSTM_MULTI_CLASS,
}
VIEWS = ("word_tfidf", "character_tfidf", "important_words", "style")


@dataclass(frozen=True)
class TextGroupRecord:
    text: str
    semantic_text: str
    finance_text: str
    style: np.ndarray
    timestamp: float
    quarter: str
    label: int


@dataclass(frozen=True)
class TextQuarterTarget:
    company: str
    quarter: str
    label: int


@dataclass
class CompanyPureTextData:
    name: str
    records: list
    run_indices: dict
    targets: dict


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(COMPANIES))
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=128)
    parser.add_argument("--word-max-features", type=int, default=50000)
    parser.add_argument("--character-max-features", type=int, default=60000)
    parser.add_argument("--important-word-max-features", type=int, default=40000)
    parser.add_argument("--regularizations", default="0.25,1,4")
    parser.add_argument("--temperatures", default="0.5,1,2")
    parser.add_argument("--fusion-step", type=float, default=0.25)
    parser.add_argument("--important-word-count", type=int, default=25)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output", default="../../output/pure_text_quarter_results.json")
    return parser.parse_args()


def run_seeds(args):
    return [args.seed + run * 100000 for run in range(args.runs)]


def build_company_data(name, company_index, prediction_path, args):
    print("%s: reading pure-text tweet groups" % name)
    frame = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["writer", "post_date", "body", "class"],
    )
    frame["body"] = frame["body"].fillna("")
    frame["writer"] = frame["writer"].fillna("<missing>")
    frame, all_groups = build_quarter_groups(
        frame, group_size=prediction_path.getTweetGroupSize())
    allowed_quarters = sorted({
        group.quarter for group in all_groups
        if 2015 <= int(group.quarter[:4]) <= args.last_test_year
    })
    targets = {}
    for group in all_groups:
        if group.quarter not in allowed_quarters:
            continue
        if group.quarter in targets and targets[group.quarter].label != group.label:
            raise ValueError("%s %s contains inconsistent targets" % (name, group.quarter))
        targets[group.quarter] = TextQuarterTarget(name, group.quarter, group.label)

    selected_by_seed = {
        seed: select_balanced_quarter_groups(
            all_groups,
            allowed_quarters,
            args.groups_per_quarter,
            seed=seed + company_index * 1000,
        )
        for seed in run_seeds(args)
    }
    records, key_to_index, selected_indices = [], {}, {}
    analyzer = SentimentIntensityAnalyzer()
    for seed, selected_groups in selected_by_seed.items():
        selected_indices[seed] = []
        for group in selected_groups:
            key = group.row_indexes
            if key not in key_to_index:
                values = frame.loc[list(group.row_indexes)]
                bodies = values["body"].astype(str).tolist()
                key_to_index[key] = len(records)
                records.append(TextGroupRecord(
                    text=" <SEP> ".join(bodies),
                    semantic_text=normalize_semantic_text(" <SEP> ".join(bodies)),
                    finance_text=finance_relevant_semantic_text(bodies),
                    style=(
                        group_style_features(
                            bodies,
                            values["writer"].tolist(),
                            values["post_date"].tolist(),
                            analyzer=analyzer,
                        )
                        if getattr(args, "include_style", True)
                        else np.zeros(len(STYLE_FEATURE_NAMES), dtype=np.float32)
                    ),
                    timestamp=float(values["post_date"].mean()),
                    quarter=group.quarter,
                    label=group.label,
                ))
            selected_indices[seed].append(key_to_index[key])
    print("%s: prepared %d unique text groups for %d runs"
          % (name, len(records), args.runs))
    del frame, all_groups, selected_by_seed
    return CompanyPureTextData(name, records, selected_indices, targets)


def selected_records(data, seed, quarters):
    quarter_set = set(quarters)
    return [
        data.records[index] for index in data.run_indices[seed]
        if data.records[index].quarter in quarter_set
    ]


def record_arrays(records):
    return {
        "texts": [record.text for record in records],
        "semantic_texts": [record.semantic_text for record in records],
        "finance_texts": [record.finance_text for record in records],
        "style": np.asarray([record.style for record in records], dtype=np.float32),
        "timestamps": np.asarray([record.timestamp for record in records], dtype=np.float64),
        "labels": np.asarray([record.label for record in records], dtype=np.int64),
        "quarters": [record.quarter for record in records],
    }


def fit_candidate_views(data, seed, train_quarters, evaluation_quarters, args):
    train = record_arrays(selected_records(data, seed, train_quarters))
    evaluation = record_arrays(selected_records(data, seed, evaluation_quarters))
    regularizations = [
        float(value) for value in args.regularizations.split(",") if value.strip()]
    temperatures = [
        float(value) for value in args.temperatures.split(",") if value.strip()]
    candidates = {view: {} for view in VIEWS}

    for view, vectorizer in (
        ("word_tfidf", word_vectorizer(args.word_max_features)),
        ("character_tfidf", character_vectorizer(args.character_max_features)),
    ):
        train_features = vectorizer.fit_transform(train["texts"])
        evaluation_features = vectorizer.transform(evaluation["texts"])
        for regularization in regularizations:
            classifier = fit_logistic_classifier(
                train_features, train["labels"], regularization, seed)
            group_probabilities = mapped_probabilities(classifier, evaluation_features)
            candidates[view][regularization] = aggregate_quarter_probabilities(
                evaluation["quarters"], group_probabilities, evaluation_quarters)

    lexicon = ImportantWordLogOdds(
        max_features=args.important_word_max_features).fit(
            train["texts"], train["labels"])
    for temperature in temperatures:
        candidates["important_words"][temperature] = aggregate_quarter_probabilities(
            evaluation["quarters"],
            lexicon.predict_proba(evaluation["texts"], temperature=temperature),
            evaluation_quarters,
        )

    for regularization in regularizations:
        classifier = fit_style_classifier(
            train["style"], train["labels"], regularization, seed)
        candidates["style"][regularization] = aggregate_quarter_probabilities(
            evaluation["quarters"],
            style_probabilities(classifier, evaluation["style"]),
            evaluation_quarters,
        )
    return candidates


def fit_selected_views(data, seed, train_quarters, test_quarters, selected, args):
    train = record_arrays(selected_records(data, seed, train_quarters))
    test = record_arrays(selected_records(data, seed, test_quarters))
    result = {}
    for view, vectorizer in (
        ("word_tfidf", word_vectorizer(args.word_max_features)),
        ("character_tfidf", character_vectorizer(args.character_max_features)),
    ):
        train_features = vectorizer.fit_transform(train["texts"])
        test_features = vectorizer.transform(test["texts"])
        classifier = fit_logistic_classifier(
            train_features, train["labels"], selected[view], seed)
        result[view] = aggregate_quarter_probabilities(
            test["quarters"], mapped_probabilities(classifier, test_features), test_quarters)

    lexicon = ImportantWordLogOdds(
        max_features=args.important_word_max_features).fit(
            train["texts"], train["labels"])
    result["important_words"] = aggregate_quarter_probabilities(
        test["quarters"],
        lexicon.predict_proba(test["texts"], temperature=selected["important_words"]),
        test_quarters,
    )
    classifier = fit_style_classifier(
        train["style"], train["labels"], selected["style"], seed)
    result["style"] = aggregate_quarter_probabilities(
        test["quarters"], style_probabilities(classifier, test["style"]), test_quarters)
    return result, lexicon.top_words(args.important_word_count)


def targets_for(company_data, quarters):
    return [
        data.targets[quarter]
        for data in company_data
        for quarter in quarters
        if quarter in data.targets
    ]


def probability_metrics(targets, probabilities):
    labels = np.asarray([target.label for target in targets], dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(4))),
        "true": labels.tolist(),
        "predicted": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "companies": [target.company for target in targets],
        "quarters": [target.quarter for target in targets],
    }
    result["per_company"] = {}
    for company in sorted({target.company for target in targets}):
        indexes = [index for index, target in enumerate(targets) if target.company == company]
        result["per_company"][company] = {
            "accuracy": float(accuracy_score(labels[indexes], predictions[indexes])),
            "mcc": float(matthews_corrcoef(labels[indexes], predictions[indexes])),
        }
    return result


def metric_ranking(targets, probabilities):
    metrics = probability_metrics(targets, probabilities)
    return (metrics["mcc"], metrics["accuracy"], -metrics["log_loss"]), metrics


def select_view_candidates(company_data, company_candidates, validation_quarters):
    targets = targets_for(company_data, validation_quarters)
    selected, diagnostics = {}, {}
    for view in VIEWS:
        candidate_values = sorted(next(iter(company_candidates.values()))[view])
        best = None
        for candidate in candidate_values:
            probabilities = np.concatenate([
                company_candidates[data.name][view][candidate]
                for data in company_data
            ])
            ranking, metrics = metric_ranking(targets, probabilities)
            if best is None or ranking > best[0]:
                best = ranking, candidate, metrics
        selected[view] = float(best[1])
        diagnostics[view] = best[2]
    return selected, diagnostics


def fusion_weight_grid(step):
    units = int(round(1.0 / step))
    if units < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("fusion-step must divide 1.0 exactly")
    for values in itertools.product(range(units + 1), repeat=len(VIEWS)):
        if sum(values) == units:
            yield np.asarray(values, dtype=np.float64) / units


def fuse_probabilities(view_probabilities, weights):
    fused = sum(
        weight * view_probabilities[view]
        for view, weight in zip(VIEWS, weights)
    )
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)


def select_fusion_weights(validation_targets, validation_probabilities, step):
    best = None
    for weights in fusion_weight_grid(step):
        probabilities = fuse_probabilities(validation_probabilities, weights)
        ranking, metrics = metric_ranking(validation_targets, probabilities)
        simplicity = -int(np.count_nonzero(weights))
        candidate_ranking = ranking + (simplicity,)
        if best is None or candidate_ranking > best[0]:
            best = candidate_ranking, weights, metrics
    return best[1], best[2]


def shuffle_quarter_probabilities(probabilities, targets, seed):
    result = probabilities.copy()
    random = np.random.RandomState(seed)
    for company in sorted({target.company for target in targets}):
        indexes = [index for index, target in enumerate(targets) if target.company == company]
        if len(indexes) < 2:
            continue
        shift = int(random.randint(1, len(indexes)))
        source = indexes[shift:] + indexes[:shift]
        result[indexes] = probabilities[source]
    return result


def past_majority_probabilities(company_data, train_quarters, test_quarters):
    result = []
    for data in company_data:
        labels = [data.targets[quarter].label for quarter in train_quarters]
        majority = int(np.bincount(labels, minlength=4).argmax())
        for _ in test_quarters:
            probability = np.zeros(4, dtype=np.float64)
            probability[majority] = 1.0
            result.append(probability)
    return np.asarray(result)


def rolling_fold(company_data, test_year, run_seed, args):
    validation_year = test_year - 1
    all_quarters = sorted({quarter for data in company_data for quarter in data.targets})
    train_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) < validation_year]
    validation_quarters = [
        quarter for quarter in all_quarters if int(quarter[:4]) == validation_year]
    test_quarters = [quarter for quarter in all_quarters if int(quarter[:4]) == test_year]
    print("  pure-text selection on %d, test on %d" % (validation_year, test_year))

    candidate_results = {
        data.name: fit_candidate_views(
            data, run_seed, train_quarters, validation_quarters, args)
        for data in company_data
    }
    selected, view_validation_metrics = select_view_candidates(
        company_data, candidate_results, validation_quarters)
    validation_targets = targets_for(company_data, validation_quarters)
    validation_probabilities = {
        view: np.concatenate([
            candidate_results[data.name][view][selected[view]]
            for data in company_data
        ])
        for view in VIEWS
    }
    fusion_weights, fusion_validation_metrics = select_fusion_weights(
        validation_targets, validation_probabilities, args.fusion_step)

    combined_quarters = train_quarters + validation_quarters
    company_test_results, important_words = {}, {}
    for data in company_data:
        company_test_results[data.name], important_words[data.name] = fit_selected_views(
            data, run_seed, combined_quarters, test_quarters, selected, args)
    test_targets = targets_for(company_data, test_quarters)
    predictions = {
        view: np.concatenate([
            company_test_results[data.name][view] for data in company_data])
        for view in VIEWS
    }
    predictions["ensemble"] = fuse_probabilities(predictions, fusion_weights)
    predictions["ensemble_shuffled"] = shuffle_quarter_probabilities(
        predictions["ensemble"], test_targets, run_seed + test_year + 50000)
    predictions["past_majority"] = past_majority_probabilities(
        company_data, combined_quarters, test_quarters)
    return {
        "test_year": test_year,
        "train_quarters": train_quarters,
        "validation_quarters": validation_quarters,
        "test_quarters": test_quarters,
        "selected_candidates": selected,
        "selected_fusion_weights": {
            view: float(weight) for view, weight in zip(VIEWS, fusion_weights)},
        "validation_metrics": {
            **view_validation_metrics,
            "ensemble": fusion_validation_metrics,
        },
        "important_words": important_words,
        "targets": test_targets,
        "predictions": predictions,
    }


def main():
    args = parse_arguments()
    company_names = [value.strip() for value in args.companies.split(",") if value.strip()]
    for name in company_names:
        if name not in COMPANIES:
            raise ValueError("Unknown company %s" % name)
    if args.first_test_year > args.last_test_year:
        raise ValueError("first-test-year must not exceed last-test-year")
    print("Pure text only: no financial inputs, baseline, embeddings or CUDA model")
    company_data = [
        build_company_data(name, index, COMPANIES[name], args)
        for index, name in enumerate(company_names)
    ]
    architectures = VIEWS + ("ensemble", "ensemble_shuffled", "past_majority")
    all_targets, run_details = None, []
    run_probabilities = {architecture: [] for architecture in architectures}
    for run_seed in run_seeds(args):
        print("\n=== pure-text run seed %d ===" % run_seed)
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
                architecture: probability_metrics(
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

    ensemble_probabilities = {
        architecture: np.mean(values, axis=0)
        for architecture, values in run_probabilities.items()
    }
    metrics = {
        architecture: probability_metrics(all_targets, probabilities)
        for architecture, probabilities in ensemble_probabilities.items()
    }
    print("\n=== pure-text rolling future ensemble ===")
    for architecture, values in metrics.items():
        print("%-22s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))

    result = {
        "experiment": "pure-text quarter-aligned group ensemble",
        "runs": args.runs,
        "seeds": run_seeds(args),
        "evaluation": {
            "first_test_year": args.first_test_year,
            "last_test_year": args.last_test_year,
            "independent_unit": "company-quarter",
            "current_quarter_text": True,
            "target": "current-quarter financial four-class label already present in tweet CSV",
            "financial_inputs_or_baseline_used": False,
            "word_embeddings_used": False,
            "test_labels_used_for_training_or_selection": False,
        },
        "configuration": {
            "companies": company_names,
            "groups_per_quarter": args.groups_per_quarter,
            "regularizations": args.regularizations,
            "temperatures": args.temperatures,
            "fusion_step": args.fusion_step,
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
