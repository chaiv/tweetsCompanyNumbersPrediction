"""Quality-filtered semantic tweet-group experiment with strict future evaluation.

Groups contain only deduplicated company/financial-event tweets. Repeated promotional patterns are
removed and each author is capped within a quarter before groups are formed. Inputs remain pure
text: no financial lag, baseline, engagement count, embedding or neural teacher is used.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import trainPureTextQuarterModel as base
import trainSemanticPureTextQuarterModel as semantic
from classifier.PureTextQuarterViews import STYLE_FEATURE_NAMES
from classifier.QualityFilteredQuarterGroups import quality_filter_tweets
from classifier.QuarterAlignedDataset import (
    build_quarter_groups,
    select_balanced_quarter_groups,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(base.COMPANIES))
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=128)
    parser.add_argument("--max-author-tweets-per-quarter", type=int, default=50)
    parser.add_argument("--minimum-semantic-tokens", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--stable-max-features", type=int, default=40000)
    parser.add_argument("--regularizations", default="0.25,1,4")
    parser.add_argument("--temperatures", default="0.5,1,2")
    parser.add_argument("--important-word-count", type=int, default=25)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output", default="../../output/quality_filtered_pure_text_quarter_results.json")
    return parser.parse_args()


def build_company_data(name, company_index, prediction_path, args):
    print("%s: reading and quality-filtering event tweets" % name)
    original = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["writer", "post_date", "body", "class"],
    )
    filtered = quality_filter_tweets(
        original,
        name,
        max_author_tweets_per_quarter=args.max_author_tweets_per_quarter,
        minimum_semantic_tokens=args.minimum_semantic_tokens,
    )
    filtered, all_groups = build_quarter_groups(
        filtered, group_size=prediction_path.getTweetGroupSize())
    allowed_quarters = sorted({
        group.quarter for group in all_groups
        if 2015 <= int(group.quarter[:4]) <= args.last_test_year
    })
    expected_quarters = [
        "%dQ%d" % (year, quarter)
        for year in range(2015, args.last_test_year + 1)
        for quarter in range(1, 5)
    ]
    missing = sorted(set(expected_quarters).difference(allowed_quarters))
    if missing:
        raise ValueError("%s quality filter removed all complete groups for %s" % (name, missing))

    targets = {}
    for group in all_groups:
        if group.quarter in allowed_quarters:
            targets[group.quarter] = base.TextQuarterTarget(
                name, group.quarter, group.label)
    selected_by_seed = {
        seed: select_balanced_quarter_groups(
            all_groups,
            allowed_quarters,
            args.groups_per_quarter,
            seed=seed + company_index * 1000,
        )
        for seed in base.run_seeds(args)
    }
    records, key_to_index, selected_indices = [], {}, {}
    for seed, selected_groups in selected_by_seed.items():
        selected_indices[seed] = []
        for group in selected_groups:
            key = group.row_indexes
            if key not in key_to_index:
                values = filtered.loc[list(group.row_indexes)]
                bodies = values["body"].astype(str).tolist()
                semantic_text = " <SEP> ".join(values["semantic_text"].astype(str).tolist())
                key_to_index[key] = len(records)
                records.append(base.TextGroupRecord(
                    text=" <SEP> ".join(bodies),
                    semantic_text=semantic_text,
                    finance_text=semantic_text,
                    style=np.zeros(len(STYLE_FEATURE_NAMES), dtype=np.float32),
                    timestamp=float(values["post_date"].mean()),
                    quarter=group.quarter,
                    label=group.label,
                ))
            selected_indices[seed].append(key_to_index[key])
    tweet_counts = filtered.groupby("reporting_quarter").size().to_dict()
    group_counts = {
        quarter: sum(group.quarter == quarter for group in all_groups)
        for quarter in allowed_quarters
    }
    quality_stats = {
        "original_tweets": int(len(original)),
        "filtered_tweets": int(len(filtered)),
        "retained_fraction": float(len(filtered) / max(len(original), 1)),
        "tweets_per_quarter": {key: int(value) for key, value in tweet_counts.items()},
        "available_groups_per_quarter": {
            key: int(value) for key, value in group_counts.items()},
        "minimum_groups_in_quarter": int(min(group_counts.values())),
    }
    print("%s: retained %d/%d tweets; %d-%d groups/quarter; %d unique selected groups"
          % (name, len(filtered), len(original), min(group_counts.values()),
             max(group_counts.values()), len(records)))
    result = base.CompanyPureTextData(name, records, selected_indices, targets)
    result.quality_stats = quality_stats
    del original, filtered, all_groups, selected_by_seed
    return result


def main():
    args = parse_arguments()
    company_names = [value.strip() for value in args.companies.split(",") if value.strip()]
    for name in company_names:
        if name not in base.COMPANIES:
            raise ValueError("Unknown company %s" % name)
    print("Quality-filtered semantic pure text: no finance, embeddings or engagement counts")
    company_data = [
        build_company_data(name, index, base.COMPANIES[name], args)
        for index, name in enumerate(company_names)
    ]
    architectures = semantic.VIEWS + (
        "validation_selected_view", "equal_ensemble", "selected_view_shuffled",
        "past_majority")
    all_targets, run_details = None, []
    run_probabilities = {architecture: [] for architecture in architectures}
    for run_seed in base.run_seeds(args):
        print("\n=== quality-filtered pure-text run seed %d ===" % run_seed)
        targets, folds = [], []
        probabilities = {architecture: [] for architecture in architectures}
        for test_year in range(args.first_test_year, args.last_test_year + 1):
            fold = semantic.rolling_fold(company_data, test_year, run_seed, args)
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
    print("\n=== quality-filtered rolling future ensemble ===")
    for architecture, values in metrics.items():
        print("%-26s accuracy %.4f mcc %.4f log_loss %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["log_loss"]))

    result = {
        "experiment": "quality-filtered semantic pure-text quarter groups",
        "runs": args.runs,
        "seeds": base.run_seeds(args),
        "evaluation": {
            "first_test_year": args.first_test_year,
            "last_test_year": args.last_test_year,
            "independent_unit": "company-quarter",
            "financial_inputs_or_baseline_used": False,
            "word_embeddings_used": False,
            "engagement_counts_used": False,
            "test_labels_used_for_training_or_selection": False,
        },
        "configuration": {
            "companies": company_names,
            "groups_per_quarter": args.groups_per_quarter,
            "max_author_tweets_per_quarter": args.max_author_tweets_per_quarter,
            "minimum_semantic_tokens": args.minimum_semantic_tokens,
            "regularizations": args.regularizations,
            "temperatures": args.temperatures,
        },
        "quality_stats": {
            data.name: data.quality_stats for data in company_data},
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
