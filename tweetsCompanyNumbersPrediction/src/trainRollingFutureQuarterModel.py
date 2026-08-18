"""Rolling-origin evaluation where every test year strictly follows train and validation."""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef

from classifier.QuarterSequenceDataset import FINANCIAL_FEATURE_NAMES
from classifier.QuarterSequenceModel import ARCHITECTURE_VIEWS
from trainQuarterSequenceModel import (
    COMPANIES,
    baseline_metrics,
    build_records,
    fit_with_validation,
    metric_summary,
    refit_and_test,
    split_records,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(COMPANIES))
    parser.add_argument("--architectures", default="financial,fusion")
    parser.add_argument("--test-years", default="2017,2018,2019")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--tweets-per-bin", type=int, default=8)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sentence-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cache", default="../../output/rolling_quarter_text_cache.npz")
    parser.add_argument("--output", default="../../output/rolling_future_quarter_results.json")
    return parser.parse_args()


def aggregate_baselines(fold_baselines):
    results = {}
    for name in fold_baselines[0]:
        true = sum((fold[name]["true"] for fold in fold_baselines), [])
        predicted = sum((fold[name]["predicted"] for fold in fold_baselines), [])
        results[name] = {
            "accuracy": float(accuracy_score(true, predicted)),
            "mcc": float(matthews_corrcoef(true, predicted)),
            "true": true,
            "predicted": predicted,
        }
    return results


def main():
    args = parse_arguments()
    company_names = [value.strip() for value in args.companies.split(",") if value.strip()]
    architectures = [value.strip() for value in args.architectures.split(",") if value.strip()]
    test_years = [int(value.strip()) for value in args.test_years.split(",") if value.strip()]
    for company in company_names:
        if company not in COMPANIES:
            raise ValueError("Unknown company %s" % company)
    for architecture in architectures:
        if architecture not in ARCHITECTURE_VIEWS:
            raise ValueError("Unknown architecture %s" % architecture)
    if min(test_years) < 2017:
        raise ValueError("Neural rolling folds need a prior training year and validation year")

    # build_records uses these fields for cache identity and the loaded year range.
    args.test_year = max(test_years)
    args.validation_year = args.test_year - 1
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)
    print("Protocol: rolling future years", test_years)
    records = build_records(company_names, args, device)

    folds = []
    fold_baselines = []
    probabilities_by_architecture = {
        architecture: [[] for _ in range(args.runs)] for architecture in architectures}
    all_test_records = []
    for test_year in test_years:
        validation_year = test_year - 1
        train_records, validation_records, test_records = split_records(
            records, validation_year, test_year)
        print("\n=== Future fold %d: train <= %d, validation %d, test %d ==="
              % (test_year, validation_year - 1, validation_year, test_year))
        print("Independent company-quarters: train %d, validation %d, test %d"
              % (len(train_records), len(validation_records), len(test_records)))
        fold_baseline = baseline_metrics(train_records + validation_records, test_records)
        fold_baselines.append(fold_baseline)
        fold_result = {"test_year": test_year, "baselines": fold_baseline, "models": {}}
        all_test_records.extend(test_records)

        for architecture in architectures:
            print("\n---", architecture, "---")
            fold_probabilities = []
            run_results = []
            for run in range(args.runs):
                run_seed = args.seed + run * 100 + test_year
                print("Run %d/%d, seed %d" % (run + 1, args.runs, run_seed))
                best_epoch, validation_loss = fit_with_validation(
                    architecture, train_records, validation_records, args, device, run_seed)
                metrics, probabilities = refit_and_test(
                    architecture, train_records, validation_records, test_records,
                    best_epoch, args, device, run_seed)
                metrics["seed"] = run_seed
                metrics["selected_epoch"] = best_epoch
                metrics["validation_loss"] = validation_loss
                run_results.append(metrics)
                fold_probabilities.append(probabilities)
                probabilities_by_architecture[architecture][run].append(probabilities)
                print("Future %d accuracy %.4f, MCC %.4f"
                      % (test_year, metrics["accuracy"], metrics["mcc"]))
            ensemble = metric_summary(test_records, np.mean(fold_probabilities, axis=0))
            fold_result["models"][architecture] = {
                "ensemble": ensemble,
                "runs": run_results,
            }
            print("Fold ensemble accuracy %.4f, MCC %.4f"
                  % (ensemble["accuracy"], ensemble["mcc"]))
        folds.append(fold_result)

    rolling_baselines = aggregate_baselines(fold_baselines)
    models = {}
    for architecture in architectures:
        run_summaries, concatenated_probabilities = [], []
        for run in range(args.runs):
            probabilities = np.concatenate(probabilities_by_architecture[architecture][run], axis=0)
            concatenated_probabilities.append(probabilities)
            run_summaries.append(metric_summary(all_test_records, probabilities))
        ensemble = metric_summary(all_test_records, np.mean(concatenated_probabilities, axis=0))
        models[architecture] = {
            "accuracy_mean": float(np.mean([value["accuracy"] for value in run_summaries])),
            "accuracy_std": float(np.std([value["accuracy"] for value in run_summaries])),
            "mcc_mean": float(np.mean([value["mcc"] for value in run_summaries])),
            "mcc_std": float(np.std([value["mcc"] for value in run_summaries])),
            "ensemble": ensemble,
            "runs": run_summaries,
        }
        print("\nROLLING %s: ensemble accuracy %.4f, MCC %.4f"
              % (architecture, ensemble["accuracy"], ensemble["mcc"]))
        for company, metrics in ensemble["per_company"].items():
            print("  %s accuracy %.4f, MCC %.4f"
                  % (company, metrics["accuracy"], metrics["mcc"]))

    results = {
        "protocol": "rolling-origin; every test year strictly follows train and validation",
        "target": "current-quarter four-class percent_change",
        "text_horizon": "current-quarter tweets available before financial result (nowcast)",
        "data_scope": "local tweet and quarterly financial CSV files only",
        "test_years": test_years,
        "independent_test_company_quarters": len(all_test_records),
        "financial_features": list(FINANCIAL_FEATURE_NAMES),
        "baselines": rolling_baselines,
        "models": models,
        "folds": folds,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
