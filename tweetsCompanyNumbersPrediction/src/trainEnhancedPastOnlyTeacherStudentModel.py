"""Enhanced past-only text experiment with quarter recognition, metadata, PCA and gating.

The target remains the current quarter's four-class financial percentage change.  Quarter IDs are
used only as an auxiliary representation-learning task on past quarters; author, volume and timing
metadata are derived from the same local tweet CSV and never from engagement counters.  PCA size,
ridge regularisation and the residual gate are selected on the immediately preceding validation
year, then evaluated on a later untouched year.
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import trainPastOnlyTeacherStudentModel as base
from classifier.PastOnlyMultitaskTeacher import PastOnlyMultitaskTeacher
from classifier.PastOnlyTeacherFeatures import TEMPORAL_FEATURE_NAMES
from classifier.QuarterMetadataFeatures import (
    TEMPORAL_METADATA_FEATURE_NAMES,
    build_quarter_metadata,
    temporal_metadata_features,
)
from classifier.QuarterSequenceDataset import percent_change_class


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(base.COMPANIES))
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--adapter-size", type=int, default=32)
    parser.add_argument("--quarter-loss-weight", type=float, default=0.75)
    parser.add_argument("--ridge-alphas", default="10,100,1000,10000")
    parser.add_argument("--pca-components", default="2,4,8")
    parser.add_argument("--residual-gates", default="0,0.25,0.5,1")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output", default="../../output/enhanced_past_only_teacher_student_results.json")
    return parser.parse_args()


def build_company_data(name, company_index, prediction_path, args):
    data = base.build_company_data(name, company_index, prediction_path, args)
    print("%s: aggregating target-independent author/volume metadata" % name)
    metadata_frame = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["writer", "post_date", "body"],
    )
    quarter_metadata = build_quarter_metadata(metadata_frame)
    data.temporal_metadata = {
        quarter: temporal_metadata_features(quarter_metadata, quarter)
        for quarter in data.targets
    }
    del metadata_frame, quarter_metadata
    return data


def create_teacher(data, train_quarters, args, device):
    quarter_to_index = {
        quarter: index for index, quarter in enumerate(sorted(train_quarters))
    }
    model = PastOnlyMultitaskTeacher(
        emb_size=data.vectors.shape[1],
        word_vectors=SimpleNamespace(vectors=data.vectors),
        num_financial_classes=base.NUM_CLASSES,
        num_training_quarters=len(quarter_to_index),
        pad_token_index=data.pad_index,
        adapter_size=args.adapter_size,
    ).to(device)
    return model, quarter_to_index


def train_teacher_epoch(model, loader, optimizer, financial_loss_function,
                        quarter_to_index, quarter_loss_weight, device):
    model.train()
    total_losses, financial_losses, quarter_losses = [], [], []
    quarter_true, quarter_predicted = [], []
    for sequences, financial_labels, quarters in loader:
        sequences = sequences.to(device, non_blocking=True)
        financial_labels = financial_labels.to(device, non_blocking=True)
        quarter_labels = torch.as_tensor(
            [quarter_to_index[quarter] for quarter in quarters],
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model.all_outputs(sequences)
        financial_loss = financial_loss_function(outputs["financial"], financial_labels)
        quarter_loss = torch.nn.functional.cross_entropy(outputs["quarter"], quarter_labels)
        loss = financial_loss + quarter_loss_weight * quarter_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        total_losses.append(float(loss.detach().cpu()))
        financial_losses.append(float(financial_loss.detach().cpu()))
        quarter_losses.append(float(quarter_loss.detach().cpu()))
        quarter_true.extend(quarter_labels.cpu().tolist())
        quarter_predicted.extend(outputs["quarter"].argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": float(np.mean(total_losses)),
        "financial_loss": float(np.mean(financial_losses)),
        "quarter_loss": float(np.mean(quarter_losses)),
        "quarter_accuracy": float(accuracy_score(quarter_true, quarter_predicted)),
    }


def teacher_optimizer(model, args):
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def fit_teacher_with_validation(data, train_quarters, validation_quarters, args, device, seed):
    base.seed_everything(seed)
    model, quarter_to_index = create_teacher(data, train_quarters, args, device)
    financial_loss_function = torch.nn.CrossEntropyLoss(
        weight=base.class_weights(data, train_quarters, device), label_smoothing=0.02)
    optimizer = teacher_optimizer(model, args)
    train_loader = base.make_loader(data, train_quarters, args.batch_size, shuffle=True)
    validation_loader = base.make_loader(
        data, validation_quarters, args.batch_size, shuffle=False)
    best_loss, best_epoch, best_state, best_metrics, stale = float("inf"), 1, None, None, 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_teacher_epoch(
            model, train_loader, optimizer, financial_loss_function,
            quarter_to_index, args.quarter_loss_weight, device)
        validation_metrics = base.evaluate_teacher(model, validation_loader, device)
        print("    enhanced teacher %s epoch %02d fin %.4f qtr %.4f qacc %.3f "
              "val %.4f acc %.3f mcc %.3f"
              % (data.name, epoch, train_metrics["financial_loss"],
                 train_metrics["quarter_loss"], train_metrics["quarter_accuracy"],
                 validation_metrics["loss"], validation_metrics["accuracy"],
                 validation_metrics["mcc"]))
        if validation_metrics["loss"] < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            best_metrics = dict(validation_metrics)
            best_metrics["training_quarter_accuracy"] = train_metrics["quarter_accuracy"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics


def refit_teacher(data, quarters, epochs, args, device, seed):
    base.seed_everything(seed)
    model, quarter_to_index = create_teacher(data, quarters, args, device)
    financial_loss_function = torch.nn.CrossEntropyLoss(
        weight=base.class_weights(data, quarters, device), label_smoothing=0.02)
    optimizer = teacher_optimizer(model, args)
    loader = base.make_loader(data, quarters, args.batch_size, shuffle=True)
    final_metrics = None
    for epoch in range(1, epochs + 1):
        final_metrics = train_teacher_epoch(
            model, loader, optimizer, financial_loss_function,
            quarter_to_index, args.quarter_loss_weight, device)
        print("    enhanced teacher %s refit %02d/%02d fin %.4f qacc %.3f"
              % (data.name, epoch, epochs, final_metrics["financial_loss"],
                 final_metrics["quarter_accuracy"]))
    return model, final_metrics


def feature_blocks(company_data, teacher_features, quarters):
    text_rows, metadata_rows, companies, targets = [], [], [], []
    num_companies = len(company_data)
    for data in company_data:
        for quarter in quarters:
            if quarter not in data.targets or (data.name, quarter) not in teacher_features:
                continue
            text_rows.append(teacher_features[(data.name, quarter)])
            metadata_rows.append(data.temporal_metadata[quarter])
            company = np.zeros(num_companies, dtype=np.float32)
            company[data.company_index] = 1.0
            companies.append(company)
            targets.append(data.targets[quarter])
    return {
        "text": np.asarray(text_rows, dtype=np.float32),
        "metadata": np.asarray(metadata_rows, dtype=np.float32),
        "company": np.asarray(companies, dtype=np.float32),
        "targets": targets,
    }


def variant_matrix(blocks, variant):
    if variant == "text":
        signal = blocks["text"]
    elif variant == "metadata":
        signal = blocks["metadata"]
    elif variant == "combined":
        signal = np.concatenate((blocks["text"], blocks["metadata"]), axis=1)
    else:
        raise ValueError("Unknown feature variant %s" % variant)
    return np.concatenate((signal, blocks["company"]), axis=1), signal.shape[1]


def pca_ridge_model(alpha, components, signal_feature_count):
    signal_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, svd_solver="full")),
    ])
    preprocessing = ColumnTransformer([
        ("signal", signal_pipeline, slice(0, signal_feature_count)),
        ("company", "passthrough", slice(signal_feature_count, None)),
    ])
    return Pipeline([
        ("features", preprocessing),
        ("ridge", Ridge(alpha=alpha)),
    ])


def prediction_metrics(targets, prediction):
    labels = np.asarray([target.label for target in targets])
    actual = np.asarray([target.percent_change for target in targets])
    classes = np.asarray([percent_change_class(value) for value in prediction])
    return {
        "accuracy": float(accuracy_score(labels, classes)),
        "mcc": float(matthews_corrcoef(labels, classes)),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def tune_model(train_x, train_targets, validation_x, validation_targets,
               signal_feature_count, alphas, component_candidates, residual_gates):
    train_actual = np.asarray([target.percent_change for target in train_targets])
    validation_baseline = np.asarray([
        target.financial_baseline for target in validation_targets])
    train_baseline = np.asarray([target.financial_baseline for target in train_targets])
    residual = residual_gates is not None
    fit_target = train_actual - train_baseline if residual else train_actual
    gates = residual_gates if residual else [1.0]
    best = None
    max_components = min(len(train_targets) - 1, signal_feature_count)
    for components in component_candidates:
        if components > max_components:
            continue
        for alpha in alphas:
            model = pca_ridge_model(alpha, components, signal_feature_count)
            model.fit(train_x, fit_target)
            raw_prediction = model.predict(validation_x)
            for gate in gates:
                prediction = (validation_baseline + gate * raw_prediction
                              if residual else raw_prediction)
                metrics = prediction_metrics(validation_targets, prediction)
                ranking = (
                    metrics["mcc"],
                    metrics["accuracy"],
                    -metrics["mae"],
                    -components,
                    -alpha,
                    -gate,
                )
                if best is None or ranking > best[0]:
                    best = (ranking, {
                        "alpha": float(alpha),
                        "pca_components": int(components),
                        "gate": float(gate),
                        "validation": metrics,
                    })
    if best is None:
        raise ValueError("No PCA component candidate fits the training fold")
    return best[1]


def fit_selected_model(train_x, train_targets, test_x, test_targets,
                       signal_feature_count, selected, residual):
    actual = np.asarray([target.percent_change for target in train_targets])
    train_baseline = np.asarray([target.financial_baseline for target in train_targets])
    test_baseline = np.asarray([target.financial_baseline for target in test_targets])
    fit_target = actual - train_baseline if residual else actual
    model = pca_ridge_model(
        selected["alpha"], selected["pca_components"], signal_feature_count)
    model.fit(train_x, fit_target)
    raw_prediction = model.predict(test_x)
    prediction = (test_baseline + selected["gate"] * raw_prediction
                  if residual else raw_prediction)
    return model, prediction


def rotate_feature_block(matrix, targets, start, stop, seed):
    result = matrix.copy()
    random_state = np.random.RandomState(seed)
    for company in sorted({target.company for target in targets}):
        indexes = [index for index, target in enumerate(targets) if target.company == company]
        if len(indexes) < 2:
            continue
        shift = int(random_state.randint(1, len(indexes)))
        source = indexes[shift:] + indexes[:shift]
        result[indexes, start:stop] = matrix[source, start:stop]
    return result


def rolling_fold(company_data, test_year, args, device, seed):
    validation_year = test_year - 1
    train_quarters = sorted({
        quarter for data in company_data for quarter in data.targets
        if int(quarter[:4]) < validation_year
    })
    validation_quarters = sorted({
        quarter for data in company_data for quarter in data.targets
        if int(quarter[:4]) == validation_year
    })
    test_quarters = sorted({
        quarter for data in company_data for quarter in data.targets
        if int(quarter[:4]) == test_year
    })
    if not train_quarters or not validation_quarters or not test_quarters:
        raise ValueError("Fold %d has an empty train, validation or test split" % test_year)

    validation_features, selected_epochs, teacher_diagnostics = {}, {}, {}
    print("  selecting enhanced teachers and PCA/gate hyperparameters")
    for offset, data in enumerate(company_data):
        company_train = [quarter for quarter in train_quarters if quarter in data.targets]
        company_validation = [
            quarter for quarter in validation_quarters if quarter in data.targets]
        model, epoch, diagnostics = fit_teacher_with_validation(
            data, company_train, company_validation, args, device, seed + offset * 100)
        output_quarters = company_train + company_validation
        outputs = base.collect_teacher_outputs(model, data, output_quarters, args, device)
        company_features, available = base.quarter_features(outputs, company_train)
        validation_features.update({
            (data.name, quarter): feature for quarter, feature in company_features.items()
        })
        selected_epochs[data.name] = epoch
        teacher_diagnostics[data.name] = {
            "selected_epoch": epoch,
            "validation_group_accuracy": diagnostics["accuracy"],
            "validation_group_mcc": diagnostics["mcc"],
            "training_quarter_accuracy_at_selected_epoch": diagnostics[
                "training_quarter_accuracy"],
            "prototype_available": available.tolist(),
        }
        base.dispose_teacher(model, device)

    train_blocks = feature_blocks(company_data, validation_features, train_quarters)
    validation_blocks = feature_blocks(
        company_data, validation_features, validation_quarters)
    alphas = [float(value) for value in args.ridge_alphas.split(",") if value.strip()]
    components = [
        int(value) for value in args.pca_components.split(",") if value.strip()]
    gates = [float(value) for value in args.residual_gates.split(",") if value.strip()]
    selected_students = {}
    for architecture, variant, residual in (
        ("text", "text", False),
        ("metadata", "metadata", False),
        ("text_metadata", "combined", False),
        ("fusion", "combined", True),
    ):
        train_x, signal_count = variant_matrix(train_blocks, variant)
        validation_x, _ = variant_matrix(validation_blocks, variant)
        selected_students[architecture] = tune_model(
            train_x, train_blocks["targets"], validation_x,
            validation_blocks["targets"], signal_count, alphas, components,
            gates if residual else None)

    print("  refitting enhanced teachers through %d and evaluating %d"
          % (validation_year, test_year))
    final_features = {}
    combined_quarters = train_quarters + validation_quarters
    for offset, data in enumerate(company_data):
        company_combined = [quarter for quarter in combined_quarters if quarter in data.targets]
        company_test = [quarter for quarter in test_quarters if quarter in data.targets]
        model, refit_metrics = refit_teacher(
            data, company_combined, selected_epochs[data.name], args, device,
            seed + 10000 + offset * 100)
        outputs = base.collect_teacher_outputs(
            model, data, company_combined + company_test, args, device)
        company_features, available = base.quarter_features(outputs, company_combined)
        final_features.update({
            (data.name, quarter): feature for quarter, feature in company_features.items()
        })
        teacher_diagnostics[data.name]["refit_training_quarter_accuracy"] = refit_metrics[
            "quarter_accuracy"]
        teacher_diagnostics[data.name]["refit_prototype_available"] = available.tolist()
        base.dispose_teacher(model, device)

    combined_blocks = feature_blocks(company_data, final_features, combined_quarters)
    test_blocks = feature_blocks(company_data, final_features, test_quarters)
    predictions, fitted_models = {}, {}
    for architecture, variant, residual in (
        ("text", "text", False),
        ("metadata", "metadata", False),
        ("text_metadata", "combined", False),
        ("fusion", "combined", True),
    ):
        combined_x, signal_count = variant_matrix(combined_blocks, variant)
        test_x, _ = variant_matrix(test_blocks, variant)
        model, prediction = fit_selected_model(
            combined_x, combined_blocks["targets"], test_x, test_blocks["targets"],
            signal_count, selected_students[architecture], residual)
        fitted_models[architecture] = model
        predictions[architecture] = prediction

    combined_test_x, combined_signal_count = variant_matrix(test_blocks, "combined")
    text_count = len(TEMPORAL_FEATURE_NAMES)
    shuffled_text_x = rotate_feature_block(
        combined_test_x, test_blocks["targets"], 0, text_count, seed + 20000)
    shuffled_all_x = rotate_feature_block(
        combined_test_x, test_blocks["targets"], 0, combined_signal_count, seed + 30000)
    baseline = np.asarray([
        target.financial_baseline for target in test_blocks["targets"]])
    gate = selected_students["fusion"]["gate"]
    predictions["finance"] = baseline
    predictions["fusion_shuffled_text"] = (
        baseline + gate * fitted_models["fusion"].predict(shuffled_text_x))
    predictions["fusion_shuffled_all"] = (
        baseline + gate * fitted_models["fusion"].predict(shuffled_all_x))

    return {
        "test_year": test_year,
        "train_quarters": train_quarters,
        "validation_quarters": validation_quarters,
        "test_quarters": test_quarters,
        "teacher": teacher_diagnostics,
        "student_selection": selected_students,
        "targets": test_blocks["targets"],
        "predictions": predictions,
    }


def main():
    args = parse_arguments()
    company_names = [value.strip() for value in args.companies.split(",") if value.strip()]
    for name in company_names:
        if name not in base.COMPANIES:
            raise ValueError("Unknown company %s" % name)
    if args.first_test_year > args.last_test_year:
        raise ValueError("first-test-year must not exceed last-test-year")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)
    print("Enhanced teacher: frozen Top2Vec + residual adapter + past quarter-ID objective")
    company_data = [
        build_company_data(name, index, base.COMPANIES[name], args)
        for index, name in enumerate(company_names)
    ]

    architectures = (
        "finance", "text", "metadata", "text_metadata", "fusion",
        "fusion_shuffled_text", "fusion_shuffled_all")
    run_predictions = {architecture: [] for architecture in architectures}
    all_targets, run_details = None, []
    for run in range(args.runs):
        run_seed = args.seed + run * 100000
        print("\n=== enhanced run %d/%d seed %d ===" % (run + 1, args.runs, run_seed))
        targets, fold_details = [], []
        fold_predictions = {architecture: [] for architecture in architectures}
        for test_year in range(args.first_test_year, args.last_test_year + 1):
            print("\n== enhanced rolling test year %d ==" % test_year)
            fold = rolling_fold(company_data, test_year, args, device, run_seed + test_year)
            fold_targets = fold.pop("targets")
            predictions = fold.pop("predictions")
            targets.extend(fold_targets)
            for architecture in architectures:
                fold_predictions[architecture].extend(predictions[architecture].tolist())
            fold["metrics"] = {
                architecture: base.summarize_predictions(
                    fold_targets, predictions[architecture])
                for architecture in architectures
            }
            fold_details.append(fold)
        if all_targets is None:
            all_targets = targets
        elif [(target.company, target.quarter) for target in targets] != [
                (target.company, target.quarter) for target in all_targets]:
            raise AssertionError("Runs produced a different test-quarter order")
        for architecture in architectures:
            run_predictions[architecture].append(np.asarray(
                fold_predictions[architecture], dtype=np.float64))
        run_details.append({"seed": run_seed, "folds": fold_details})

    ensemble_predictions = {
        architecture: np.mean(values, axis=0)
        for architecture, values in run_predictions.items()
    }
    metrics = {
        architecture: base.summarize_predictions(all_targets, prediction)
        for architecture, prediction in ensemble_predictions.items()
    }
    print("\n=== enhanced rolling future ensemble ===")
    for architecture, values in metrics.items():
        print("%-24s accuracy %.4f mcc %.4f mae %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["mae"]))

    result = {
        "experiment": "enhanced past-only LSTM teacher, metadata, PCA and residual gate",
        "device": str(device),
        "runs": args.runs,
        "seeds": [args.seed + run * 100000 for run in range(args.runs)],
        "evaluation": {
            "first_test_year": args.first_test_year,
            "last_test_year": args.last_test_year,
            "independent_unit": "company-quarter",
            "current_quarter_text_and_metadata": True,
            "target": "current-quarter financial percent_change four-class bucket",
            "auxiliary_target": "past training-quarter ID only",
            "engagement_metadata_used": False,
            "test_labels_used_for_training_or_selection": False,
        },
        "configuration": {
            "companies": company_names,
            "groups_per_quarter": args.groups_per_quarter,
            "max_tokens": args.max_tokens,
            "adapter_size": args.adapter_size,
            "quarter_loss_weight": args.quarter_loss_weight,
            "pca_components": args.pca_components,
            "ridge_alphas": args.ridge_alphas,
            "residual_gates": args.residual_gates,
            "teacher_text_features": len(TEMPORAL_FEATURE_NAMES),
            "metadata_features": len(TEMPORAL_METADATA_FEATURE_NAMES),
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
