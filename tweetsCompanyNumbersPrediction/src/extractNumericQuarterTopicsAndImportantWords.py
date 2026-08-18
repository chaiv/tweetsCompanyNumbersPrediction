"""Connect the latest quarter models to dissertation step 3 explanations.

The script replays the already selected rolling-origin models, verifies that their Accuracy/MCC
match the stored evaluation, and writes privacy-preserving feature, important-word and topic
summaries.  Raw tweets, authors, handles, URLs and tweet identifiers are never serialized.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

import trainNumericTextSignalQuarterModel as model
import trainPureTextQuarterModel as base
from featureinterpretation.NumericQuarterTextExplanations import (
    PastOnlyNmfTopics,
    aggregate_linear_explanations,
    balanced_documents,
    fit_past_only_important_words,
    heldout_important_words,
    linear_class_feature_contributions,
    model_linked_cue_words,
)


REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_RESULTS = os.path.join(
    REPOSITORY_ROOT, "output", "numeric_text_signal_quarter_results.json")
DEFAULT_OUTPUT = os.path.join(
    REPOSITORY_ROOT, "output", "numeric_text_topics_important_words.json")
PRIMARY_ARCHITECTURE = "seasonal_tesla_conflict_gate"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-results", default=DEFAULT_MODEL_RESULTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--topics", type=int, default=6)
    parser.add_argument("--words-per-topic", type=int, default=10)
    parser.add_argument("--top-topics", type=int, default=3)
    parser.add_argument("--top-important-words", type=int, default=15)
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument("--max-documents-per-quarter", type=int, default=250)
    return parser.parse_args()


def _load_results(path):
    with open(os.path.abspath(path), encoding="utf-8") as input_file:
        return json.load(input_file)


def _mean(values):
    return np.mean(np.asarray(values, dtype=np.float64), axis=0)


def _rounded_probabilities(values):
    return [float(value) for value in np.asarray(values, dtype=np.float64)]


def _capped_raw_documents(documents, maximum):
    values = list(documents)
    if len(values) <= int(maximum):
        return values
    indexes = np.linspace(0, len(values) - 1, int(maximum), dtype=int)
    return [values[index] for index in indexes]


def _forbidden_output_key_audit(value, path="root"):
    forbidden = {"body", "bodies", "tweet", "tweets", "tweet_id", "writer", "author", "url"}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise AssertionError("Forbidden raw-content output key at %s.%s" % (path, key))
            _forbidden_output_key_audit(child, "%s.%s" % (path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_output_key_audit(child, "%s[%d]" % (path, index))


def replay_models(stored, company_data, top_features):
    """Replay stored selections and retain branch probabilities plus exact text logits."""
    smoothing = float(stored["configuration"]["seasonal_smoothing"])
    by_quarter = defaultdict(lambda: {
        "numeric_probabilities": [],
        "seasonal_probabilities": [],
        "forward_probabilities": [],
        "pre_gate_probabilities": [],
        "primary_probabilities": [],
        "gate_activations": [],
        "numeric_explanations": defaultdict(list),
        "selected_models": [],
    })
    run_primary, run_numeric = [], []
    all_targets = None
    fold_protocol = {}
    for run in stored["run_details"]:
        seed = int(run["seed"])
        current_targets, current_primary, current_numeric = [], [], []
        for fold in run["folds"]:
            test_year = int(fold["test_year"])
            train_quarters = list(fold["train_quarters"])
            validation_quarters = list(fold["validation_quarters"])
            test_quarters = list(fold["test_quarters"])
            past_quarters = train_quarters + validation_quarters
            train_rows = model.rows_for(company_data, past_quarters)
            test_rows = model.rows_for(company_data, test_quarters)
            selected = fold["selected"]["numeric_validation_selected"]
            fitted = model.fit_numeric_model(
                company_data,
                train_rows,
                test_rows,
                selected["feature_mode"],
                selected["metadata"],
                float(selected["regularization"]),
                seed,
            )
            numeric = fitted.probabilities
            seasonal = model.seasonal_probabilities(
                company_data, past_quarters, test_rows, smoothing)
            forward = model.forward_level_probabilities(company_data, test_rows)
            seasonal_numeric = model.fuse_probabilities(seasonal, numeric, 0.5)
            pre_gate = model.tesla_forward_fusion(
                test_rows, seasonal_numeric, forward, 0.5)
            changes = model.delivery_estimate_changes(company_data, test_rows)
            primary = model.tesla_conflict_gate(test_rows, pre_gate, numeric, changes)
            targets = base.targets_for(company_data, test_quarters)
            current_targets.extend(targets)
            current_primary.append(primary)
            current_numeric.append(numeric)
            fold_protocol[test_year] = {
                "train_quarters": train_quarters,
                "validation_quarters": validation_quarters,
                "test_quarters": test_quarters,
            }
            for row_index, ((company, quarter), target) in enumerate(zip(test_rows, targets)):
                key = (company, quarter)
                record = by_quarter[key]
                record["true_class"] = int(target.label)
                record["numeric_probabilities"].append(numeric[row_index])
                record["seasonal_probabilities"].append(seasonal[row_index])
                record["forward_probabilities"].append(forward[row_index])
                record["pre_gate_probabilities"].append(pre_gate[row_index])
                record["primary_probabilities"].append(primary[row_index])
                record["gate_activations"].append(bool(
                    not np.allclose(pre_gate[row_index], primary[row_index])))
                record["selected_models"].append({
                    "seed": seed,
                    "metadata": selected["metadata"],
                    "feature_mode": selected["feature_mode"],
                    "regularization": float(selected["regularization"]),
                })
                for class_index in range(4):
                    record["numeric_explanations"][class_index].append(
                        linear_class_feature_contributions(
                            fitted, row_index, class_index, top_n=None))
        combined_primary = np.concatenate(current_primary)
        combined_numeric = np.concatenate(current_numeric)
        run_primary.append(combined_primary)
        run_numeric.append(combined_numeric)
        if all_targets is None:
            all_targets = current_targets
        elif [(v.company, v.quarter, v.label) for v in all_targets] != [
                (v.company, v.quarter, v.label) for v in current_targets]:
            raise AssertionError("Run target order changed")
    averaged_primary = _mean(run_primary)
    averaged_numeric = _mean(run_numeric)
    primary_metrics = base.probability_metrics(all_targets, averaged_primary)
    numeric_metrics = base.probability_metrics(all_targets, averaged_numeric)
    stored_metrics = stored["metrics"][PRIMARY_ARCHITECTURE]
    predictions_match = (
        primary_metrics["predicted"] == stored_metrics["predicted"]
        and np.isclose(primary_metrics["accuracy"], stored_metrics["accuracy"])
        and np.isclose(primary_metrics["mcc"], stored_metrics["mcc"])
    )
    if not predictions_match:
        raise AssertionError("Explanation replay does not match stored primary predictions")

    explanations = {}
    for key, record in by_quarter.items():
        numeric_probabilities = _mean(record["numeric_probabilities"])
        primary_probabilities = _mean(record["primary_probabilities"])
        numeric_class = int(numeric_probabilities.argmax())
        primary_class = int(primary_probabilities.argmax())
        full_attribution = aggregate_linear_explanations(
            record["numeric_explanations"][numeric_class], top_n=1000000)
        top_attribution = {
            key: value for key, value in full_attribution.items() if key != "features"
        }
        top_attribution["features"] = full_attribution["features"][:int(top_features)]
        explanations[key] = {
            "company": key[0],
            "quarter": key[1],
            "true_class": int(record["true_class"]),
            "numeric_text_predicted_class": numeric_class,
            "primary_predicted_class": primary_class,
            "correct": bool(primary_class == int(record["true_class"])),
            "decision_path": {
                "seasonal_probabilities": _rounded_probabilities(
                    _mean(record["seasonal_probabilities"])),
                "numeric_text_probabilities": _rounded_probabilities(numeric_probabilities),
                "forward_level_probabilities": _rounded_probabilities(
                    _mean(record["forward_probabilities"])),
                "before_conflict_gate_probabilities": _rounded_probabilities(
                    _mean(record["pre_gate_probabilities"])),
                "final_probabilities": _rounded_probabilities(primary_probabilities),
                "conflict_gate_activation_fraction": float(np.mean(
                    record["gate_activations"])),
            },
            "selected_numeric_text_models": record["selected_models"],
            "numeric_text_feature_attribution": top_attribution,
            "_all_numeric_text_feature_contributions": full_attribution["features"],
        }
    return explanations, primary_metrics, numeric_metrics, fold_protocol


def attach_words_and_topics(explanations, company_data, fold_protocol, args):
    by_name = {value.name: value for value in company_data}
    catalogs = []
    for test_year in sorted(fold_protocol):
        protocol = fold_protocol[test_year]
        past_quarters = protocol["train_quarters"] + protocol["validation_quarters"]
        for company, data in by_name.items():
            labels = {quarter: target.label for quarter, target in data.targets.items()}
            lexicon = fit_past_only_important_words(
                data.relevant_bodies,
                labels,
                past_quarters,
                max_per_quarter=args.max_documents_per_quarter,
            )
            training_documents, _ = balanced_documents(
                data.relevant_bodies,
                past_quarters,
                args.max_documents_per_quarter,
            )
            topics = PastOnlyNmfTopics(
                topic_count=args.topics,
                seed=1337 + test_year,
            ).fit(training_documents)
            catalogs.append({
                "company": company,
                "test_year": test_year,
                "fit_cutoff": "%dQ4" % (test_year - 1),
                "training_document_count": int(len(training_documents)),
                "topics": topics.catalog(args.words_per_topic),
            })
            for quarter in protocol["test_quarters"]:
                key = (company, quarter)
                if key not in explanations:
                    continue
                explanation = explanations[key]
                heldout_documents = _capped_raw_documents(
                    data.relevant_bodies.get(quarter, []),
                    args.max_documents_per_quarter,
                )
                final_words = heldout_important_words(
                    lexicon,
                    heldout_documents,
                    explanation["primary_predicted_class"],
                    args.top_important_words,
                )
                numeric_words = heldout_important_words(
                    lexicon,
                    heldout_documents,
                    explanation["numeric_text_predicted_class"],
                    args.top_important_words,
                )
                cue_words = model_linked_cue_words(
                    heldout_documents,
                    company,
                    explanation.pop("_all_numeric_text_feature_contributions"),
                    args.top_important_words,
                )
                explanation["important_words"] = {
                    "model_linked_cues": cue_words,
                    "past_only_words_for_numeric_text_class": numeric_words,
                    "past_only_words_for_final_class": final_words,
                }
                explanation["topics"] = topics.describe(
                    heldout_documents,
                    important_words=final_words,
                    cue_words=cue_words,
                    top_n=args.top_topics,
                )
                explanation["interpretation_scope"] = {
                    "feature_attribution": "exact for the selected numeric-text logistic branch",
                    "model_linked_cues": (
                        "family-level bridge; exact aggregate feature contribution, descriptive "
                        "allocation to matched cue words"
                    ),
                    "important_words": (
                        "past-only quarter-stable class association present in held-out text; "
                        "not a causal contribution to the hybrid"
                    ),
                    "topics": (
                        "past-only NMF summary of held-out relevant text; contextual, not causal"
                    ),
                    "final_hybrid": (
                        "auditable probability path through seasonal prior, numeric text, "
                        "Tesla forward signal and exploratory conflict gate"
                    ),
                }
    return catalogs


def main():
    args = parse_arguments()
    stored = _load_results(args.model_results)
    companies = list(stored["configuration"]["companies"])
    last_test_year = max(
        int(fold["test_year"])
        for run in stored["run_details"]
        for fold in run["folds"]
    )
    print("Rebuilding numeric-text branches for dissertation step 3")
    company_data = [
        model.build_company_data(name, base.COMPANIES[name], last_test_year)
        for name in companies
    ]
    explanations, primary_metrics, numeric_metrics, fold_protocol = replay_models(
        stored, company_data, args.top_features)
    topic_catalogs = attach_words_and_topics(
        explanations, company_data, fold_protocol, args)
    result = {
        "experiment": "latest quarter models connected to dissertation step 3",
        "source_model_results": os.path.abspath(args.model_results),
        "primary_architecture": PRIMARY_ARCHITECTURE,
        "target": "four-class current-quarter change in quarterly numbers",
        "metrics": {
            "primary_replayed": primary_metrics,
            "numeric_text_branch_replayed": numeric_metrics,
            "primary_stored": stored["metrics"][PRIMARY_ARCHITECTURE],
        },
        "verification": {
            "stored_predictions_reproduced": True,
            "accuracy_matches": True,
            "mcc_matches": True,
            "rolling_origin_future_test_retained": True,
            "test_labels_used_for_topic_or_word_fitting": False,
        },
        "method": {
            "exact_numeric_text_attribution": "standardized feature value times OVR coefficient",
            "important_words": (
                "quarter-stable class log-odds fitted only on train plus validation quarters"
            ),
            "topics": "TF-IDF plus NMF fitted only on train plus validation quarters",
            "topic_role": "contextual summary, not an additive final-model attribution",
            "document_sampling": (
                "deterministic and balanced, at most %d relevant documents per quarter"
                % args.max_documents_per_quarter
            ),
        },
        "privacy": {
            "raw_text_serialized": False,
            "authors_serialized": False,
            "handles_serialized": False,
            "urls_serialized": False,
            "tweet_ids_serialized": False,
            "only_aggregate_counts_terms_features_and_topic_words_serialized": True,
        },
        "topic_catalogs": topic_catalogs,
        "quarter_explanations": [
            explanations[key] for key in sorted(explanations)
        ],
    }
    _forbidden_output_key_audit(result)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print("Primary Accuracy %.4f MCC %.4f" % (
        primary_metrics["accuracy"], primary_metrics["mcc"]))
    print("Numeric text Accuracy %.4f MCC %.4f" % (
        numeric_metrics["accuracy"], numeric_metrics["mcc"]))
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
