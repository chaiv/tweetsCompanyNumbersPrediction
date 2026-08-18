"""Past-only LSTM teacher and quarter-level residual student.

This experiment deliberately reuses the original Top2Vec/LSTM mechanism that can recognize the
quarter associated with tweet groups, but changes the evaluation unit and chronology:

* groups never cross quarter boundaries;
* learned teacher parameters use only labelled quarters before the evaluated year;
* group representations are aggregated to exactly one row per company-quarter;
* the quarter-level student predicts either the target change from text alone or the residual over
  a strictly lagged same-quarter-last-year financial baseline;
* Accuracy and MCC are reported only for later, independent company-quarters.

The stored Top2Vec vectors were trained unsupervised on the repository's full tweet corpus.  They
are frozen here, so future labels cannot enter through them, but this is a transductive vocabulary
initialisation rather than a fully past-only language-model pretraining procedure.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from PredictionModelPath import (
    AMAZON_REVENUE_10_LSTM_MULTI_CLASS,
    APPLE__EPS_10_LSTM_MULTI_CLASS,
    TESLA_CAR_SALES_10_LSTM_MULTI_CLASS,
)
from classifier.LSTMNN import LSTMNN, MEAN_POOLING
from classifier.PastOnlyTeacherFeatures import (
    TEMPORAL_FEATURE_NAMES,
    compute_class_prototypes,
    summarize_teacher_outputs,
    temporal_teacher_features,
)
from classifier.QuarterAlignedDataset import (
    EncodedQuarterGroupDataset,
    build_quarter_groups,
    select_balanced_quarter_groups,
)
from classifier.QuarterSequenceDataset import (
    lagged_financial_sequence,
    percent_change_class,
    prepare_financial_quarters,
)
from nlpvectors.TweetTokenizer import TweetTokenizer
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter


COMPANIES = {
    "amazon": AMAZON_REVENUE_10_LSTM_MULTI_CLASS,
    "apple": APPLE__EPS_10_LSTM_MULTI_CLASS,
    "tesla": TESLA_CAR_SALES_10_LSTM_MULTI_CLASS,
}
NUM_CLASSES = 4


@dataclass
class QuarterTarget:
    company: str
    company_index: int
    quarter: str
    label: int
    percent_change: float
    financial_baseline: float


@dataclass
class CompanyTeacherData:
    name: str
    company_index: int
    vectors: np.ndarray
    pad_index: int
    dataset: EncodedQuarterGroupDataset
    quarter_indices: dict
    targets: dict


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", default=",".join(COMPANIES))
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-test-year", type=int, default=2019)
    parser.add_argument("--groups-per-quarter", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--ridge-alphas", default="1,10,100,1000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output", default="../../output/past_only_teacher_student_results.json")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_company_data(name, company_index, prediction_path, args):
    print("%s: reading quarter-aligned tweet groups" % name)
    frame = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["post_date", "body", "class"],
    )
    frame["body"] = frame["body"].fillna("")
    frame, all_groups = build_quarter_groups(
        frame, group_size=prediction_path.getTweetGroupSize())
    allowed_quarters = sorted({
        group.quarter for group in all_groups
        if 2015 <= int(group.quarter[:4]) <= args.last_test_year
    })
    selected_groups = select_balanced_quarter_groups(
        all_groups,
        allowed_quarters,
        args.groups_per_quarter,
        seed=args.seed + company_index * 1000,
    )

    print("%s: loading frozen Top2Vec vectors and encoding %d groups"
          % (name, len(selected_groups)))
    word_vectors = KeyedVectors.load_word2vec_format(
        prediction_path.getWordVectorsPath(), binary=False)
    encoder = WordVectorsIDEncoder(word_vectors)
    dataset = EncodedQuarterGroupDataset(
        frame,
        selected_groups,
        TweetTokenizer(DefaultWordFilter()),
        encoder,
        max_tokens=args.max_tokens,
    )
    quarter_indices = {}
    for index, quarter in enumerate(dataset.quarters):
        quarter_indices.setdefault(quarter, []).append(index)

    financial = prepare_financial_quarters(
        pd.read_csv(prediction_path.getFinancialNumbersPath()))
    financial_by_quarter = financial.set_index("quarter")
    targets = {}
    for quarter in allowed_quarters:
        if quarter not in financial_by_quarter.index:
            raise ValueError("%s has tweets but no financial target for %s" % (name, quarter))
        target_row = financial_by_quarter.loc[quarter]
        if pd.isna(target_row["percent_change"]):
            continue
        tweet_labels = {dataset.labels[index] for index in quarter_indices[quarter]}
        label = int(target_row["label"])
        if tweet_labels != {label}:
            raise ValueError("%s %s tweet labels %s != financial label %d"
                             % (name, quarter, tweet_labels, label))
        financial_sequence = lagged_financial_sequence(financial, quarter, lookback=4)
        targets[quarter] = QuarterTarget(
            company=name,
            company_index=company_index,
            quarter=quarter,
            label=label,
            percent_change=float(target_row["percent_change"]),
            financial_baseline=float(financial_sequence[0, 1]) * 100.0,
        )
    vectors = np.asarray(word_vectors.vectors, dtype=np.float32)
    result = CompanyTeacherData(
        name=name,
        company_index=company_index,
        vectors=vectors,
        pad_index=encoder.getPADTokenID(),
        dataset=dataset,
        quarter_indices=quarter_indices,
        targets=targets,
    )
    del word_vectors, frame, all_groups
    return result


def indexes_for_quarters(data, quarters):
    return [index for quarter in quarters for index in data.quarter_indices.get(quarter, [])]


def make_loader(data, quarters, batch_size, shuffle):
    indexes = indexes_for_quarters(data, quarters)
    if not indexes:
        raise ValueError("%s has no groups for quarters %s" % (data.name, quarters))

    def collate(batch):
        sequences, labels, group_quarters = zip(*batch)
        return (
            pad_sequence(sequences, batch_first=True, padding_value=data.pad_index),
            torch.as_tensor(labels, dtype=torch.long),
            list(group_quarters),
        )

    return DataLoader(
        Subset(data.dataset, indexes),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )


def class_weights(data, quarters, device):
    labels = [data.dataset.labels[index] for index in indexes_for_quarters(data, quarters)]
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(labels) / np.maximum(counts, 1.0)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def create_teacher(data, device):
    return LSTMNN(
        emb_size=data.vectors.shape[1],
        word_vectors=SimpleNamespace(vectors=data.vectors),
        num_classes=NUM_CLASSES,
        pooling=MEAN_POOLING,
        padTokenIdx=data.pad_index,
        freeze_embeddings=True,
    ).to(device)


def train_teacher_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    losses = []
    for sequences, labels, _ in loader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(sequences), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_teacher(model, loader, device):
    model.eval()
    losses, labels, predictions = [], [], []
    for sequences, batch_labels, _ in loader:
        sequences = sequences.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        logits = model(sequences)
        losses.append(float(torch.nn.functional.cross_entropy(logits, batch_labels).cpu()))
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def fit_teacher_with_validation(data, train_quarters, validation_quarters, args, device, seed):
    seed_everything(seed)
    model = create_teacher(data, device)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights(data, train_quarters, device), label_smoothing=0.02)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(data, train_quarters, args.batch_size, shuffle=True)
    validation_loader = make_loader(
        data, validation_quarters, args.batch_size, shuffle=False)
    best_loss, best_epoch, best_state, best_metrics, stale = float("inf"), 1, None, None, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_teacher_epoch(
            model, train_loader, optimizer, loss_function, device)
        metrics = evaluate_teacher(model, validation_loader, device)
        print("    teacher %s epoch %02d train %.4f val %.4f acc %.3f mcc %.3f"
              % (data.name, epoch, train_loss, metrics["loss"],
                 metrics["accuracy"], metrics["mcc"]))
        if metrics["loss"] < best_loss - 1e-4:
            best_loss, best_epoch, best_metrics, stale = (
                metrics["loss"], epoch, metrics, 0)
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
    seed_everything(seed)
    model = create_teacher(data, device)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights(data, quarters, device), label_smoothing=0.02)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = make_loader(data, quarters, args.batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        loss = train_teacher_epoch(model, loader, optimizer, loss_function, device)
        print("    teacher %s refit %02d/%02d loss %.4f"
              % (data.name, epoch, epochs, loss))
    return model


@torch.no_grad()
def collect_teacher_outputs(model, data, quarters, args, device):
    model.eval()
    collected = {
        quarter: {"hidden": [], "logits": [], "labels": []} for quarter in quarters
    }
    for sequences, labels, group_quarters in make_loader(
            data, quarters, args.batch_size, shuffle=False):
        sequences = sequences.to(device, non_blocking=True)
        hidden = model.encode(sequences)
        logits = model.fc3(hidden)
        for index, quarter in enumerate(group_quarters):
            collected[quarter]["hidden"].append(hidden[index].cpu().numpy())
            collected[quarter]["logits"].append(logits[index].cpu().numpy())
            collected[quarter]["labels"].append(int(labels[index]))
    for values in collected.values():
        values["hidden"] = np.asarray(values["hidden"], dtype=np.float32)
        values["logits"] = np.asarray(values["logits"], dtype=np.float32)
        values["labels"] = np.asarray(values["labels"], dtype=np.int64)
    return collected


def quarter_features(outputs, prototype_quarters):
    prototype_hidden = np.concatenate([
        outputs[quarter]["hidden"] for quarter in prototype_quarters
    ])
    prototype_labels = np.concatenate([
        outputs[quarter]["labels"] for quarter in prototype_quarters
    ])
    prototypes, available = compute_class_prototypes(
        prototype_hidden, prototype_labels, num_classes=NUM_CLASSES)
    summaries = {
        quarter: summarize_teacher_outputs(
            values["hidden"], values["logits"], prototypes, available)
        for quarter, values in outputs.items()
    }
    return {
        quarter: temporal_teacher_features(summaries, quarter)
        for quarter in summaries
    }, available


def dispose_teacher(model, device):
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def feature_matrix(company_data, features, quarters):
    rows, targets = [], []
    num_companies = len(company_data)
    for data in company_data:
        for quarter in quarters:
            if quarter not in data.targets or (data.name, quarter) not in features:
                continue
            company_one_hot = np.zeros(num_companies, dtype=np.float32)
            company_one_hot[data.company_index] = 1.0
            rows.append(np.concatenate((features[(data.name, quarter)], company_one_hot)))
            targets.append(data.targets[quarter])
    return np.asarray(rows, dtype=np.float32), targets


def ridge_model(alpha):
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def tune_students(train_x, train_targets, validation_x, validation_targets, alphas):
    train_actual = np.asarray([target.percent_change for target in train_targets])
    train_baseline = np.asarray([target.financial_baseline for target in train_targets])
    validation_actual = np.asarray([target.percent_change for target in validation_targets])
    validation_baseline = np.asarray([
        target.financial_baseline for target in validation_targets])
    validation_labels = np.asarray([target.label for target in validation_targets])

    selected = {}
    for architecture in ("text", "fusion"):
        best = None
        for alpha in alphas:
            model = ridge_model(alpha)
            fit_target = (train_actual if architecture == "text"
                          else train_actual - train_baseline)
            model.fit(train_x, fit_target)
            prediction = model.predict(validation_x)
            if architecture == "fusion":
                prediction = validation_baseline + prediction
            classes = np.asarray([percent_change_class(value) for value in prediction])
            metrics = {
                "accuracy": float(accuracy_score(validation_labels, classes)),
                "mcc": float(matthews_corrcoef(validation_labels, classes)),
                "mae": float(mean_absolute_error(validation_actual, prediction)),
            }
            ranking = (metrics["mcc"], metrics["accuracy"], -metrics["mae"], -alpha)
            if best is None or ranking > best[0]:
                best = (ranking, float(alpha), metrics)
        selected[architecture] = {"alpha": best[1], "validation": best[2]}
    return selected


def fit_students(train_x, train_targets, test_x, test_targets, selected):
    actual = np.asarray([target.percent_change for target in train_targets])
    baseline = np.asarray([target.financial_baseline for target in train_targets])
    test_baseline = np.asarray([target.financial_baseline for target in test_targets])
    text_model = ridge_model(selected["text"]["alpha"])
    fusion_model = ridge_model(selected["fusion"]["alpha"])
    text_model.fit(train_x, actual)
    fusion_model.fit(train_x, actual - baseline)
    text_prediction = text_model.predict(test_x)
    fusion_prediction = test_baseline + fusion_model.predict(test_x)
    return text_model, fusion_model, text_prediction, fusion_prediction


def rotate_text_within_company(matrix, targets, text_feature_count, seed):
    result = matrix.copy()
    random_state = np.random.RandomState(seed)
    for company in sorted({target.company for target in targets}):
        indexes = [index for index, target in enumerate(targets) if target.company == company]
        if len(indexes) < 2:
            continue
        shift = int(random_state.randint(1, len(indexes)))
        source = indexes[shift:] + indexes[:shift]
        result[indexes, :text_feature_count] = matrix[source, :text_feature_count]
    return result


def summarize_predictions(targets, continuous_prediction):
    actual = np.asarray([target.percent_change for target in targets])
    labels = np.asarray([target.label for target in targets])
    predictions = np.asarray([
        percent_change_class(value) for value in continuous_prediction], dtype=np.int64)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "mae": float(mean_absolute_error(actual, continuous_prediction)),
        "true": labels.tolist(),
        "predicted": predictions.tolist(),
        "predicted_percent_change": np.asarray(continuous_prediction).tolist(),
        "actual_percent_change": actual.tolist(),
        "companies": [target.company for target in targets],
        "quarters": [target.quarter for target in targets],
    }
    result["per_company"] = {}
    for company in sorted({target.company for target in targets}):
        indexes = [index for index, target in enumerate(targets) if target.company == company]
        result["per_company"][company] = {
            "accuracy": float(accuracy_score(labels[indexes], predictions[indexes])),
            "mcc": float(matthews_corrcoef(labels[indexes], predictions[indexes])),
            "mae": float(mean_absolute_error(
                actual[indexes], np.asarray(continuous_prediction)[indexes])),
        }
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
    print("  selecting teacher epochs and quarter-student hyperparameters")
    for offset, data in enumerate(company_data):
        company_train = [quarter for quarter in train_quarters if quarter in data.targets]
        company_validation = [
            quarter for quarter in validation_quarters if quarter in data.targets]
        model, epoch, diagnostics = fit_teacher_with_validation(
            data,
            company_train,
            company_validation,
            args,
            device,
            seed + offset * 100,
        )
        output_quarters = company_train + company_validation
        outputs = collect_teacher_outputs(model, data, output_quarters, args, device)
        company_features, available = quarter_features(outputs, company_train)
        validation_features.update({
            (data.name, quarter): feature
            for quarter, feature in company_features.items()
        })
        selected_epochs[data.name] = epoch
        teacher_diagnostics[data.name] = {
            "selected_epoch": epoch,
            "validation_group_accuracy": diagnostics["accuracy"],
            "validation_group_mcc": diagnostics["mcc"],
            "prototype_available": available.tolist(),
        }
        dispose_teacher(model, device)

    train_x, train_targets = feature_matrix(
        company_data, validation_features, train_quarters)
    validation_x, validation_targets = feature_matrix(
        company_data, validation_features, validation_quarters)
    alphas = [float(value) for value in args.ridge_alphas.split(",") if value.strip()]
    selected_students = tune_students(
        train_x, train_targets, validation_x, validation_targets, alphas)

    print("  refitting teachers through %d and evaluating %d" % (validation_year, test_year))
    final_features = {}
    combined_quarters = train_quarters + validation_quarters
    for offset, data in enumerate(company_data):
        company_combined = [quarter for quarter in combined_quarters if quarter in data.targets]
        company_test = [quarter for quarter in test_quarters if quarter in data.targets]
        model = refit_teacher(
            data,
            company_combined,
            selected_epochs[data.name],
            args,
            device,
            seed + 10000 + offset * 100,
        )
        output_quarters = company_combined + company_test
        outputs = collect_teacher_outputs(model, data, output_quarters, args, device)
        company_features, available = quarter_features(outputs, company_combined)
        final_features.update({
            (data.name, quarter): feature
            for quarter, feature in company_features.items()
        })
        teacher_diagnostics[data.name]["refit_prototype_available"] = available.tolist()
        dispose_teacher(model, device)

    combined_x, combined_targets = feature_matrix(
        company_data, final_features, combined_quarters)
    test_x, test_targets = feature_matrix(company_data, final_features, test_quarters)
    text_model, fusion_model, text_prediction, fusion_prediction = fit_students(
        combined_x, combined_targets, test_x, test_targets, selected_students)
    shuffled_x = rotate_text_within_company(
        test_x, test_targets, len(TEMPORAL_FEATURE_NAMES), seed + 20000)
    shuffled_prediction = np.asarray([
        target.financial_baseline for target in test_targets
    ]) + fusion_model.predict(shuffled_x)
    finance_prediction = np.asarray([
        target.financial_baseline for target in test_targets])
    del text_model, fusion_model

    return {
        "test_year": test_year,
        "train_quarters": train_quarters,
        "validation_quarters": validation_quarters,
        "test_quarters": test_quarters,
        "teacher": teacher_diagnostics,
        "student_selection": selected_students,
        "targets": test_targets,
        "predictions": {
            "finance": finance_prediction,
            "text": text_prediction,
            "fusion": fusion_prediction,
            "fusion_shuffled_text": shuffled_prediction,
        },
    }


def main():
    args = parse_arguments()
    company_names = [value.strip() for value in args.companies.split(",") if value.strip()]
    for name in company_names:
        if name not in COMPANIES:
            raise ValueError("Unknown company %s" % name)
    if args.first_test_year > args.last_test_year:
        raise ValueError("first-test-year must not exceed last-test-year")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)
    print("Text teacher: original two-layer LSTM with frozen full-corpus Top2Vec vectors")
    company_data = [
        build_company_data(name, index, COMPANIES[name], args)
        for index, name in enumerate(company_names)
    ]

    all_targets = None
    run_predictions = {
        architecture: []
        for architecture in ("finance", "text", "fusion", "fusion_shuffled_text")
    }
    run_details = []
    for run in range(args.runs):
        run_seed = args.seed + run * 100000
        print("\n=== run %d/%d seed %d ===" % (run + 1, args.runs, run_seed))
        fold_details, targets = [], []
        fold_predictions = {architecture: [] for architecture in run_predictions}
        for test_year in range(args.first_test_year, args.last_test_year + 1):
            print("\n== rolling test year %d ==" % test_year)
            fold = rolling_fold(company_data, test_year, args, device, run_seed + test_year)
            fold_targets = fold.pop("targets")
            predictions = fold.pop("predictions")
            targets.extend(fold_targets)
            for architecture in fold_predictions:
                fold_predictions[architecture].extend(predictions[architecture].tolist())
            fold["metrics"] = {
                architecture: summarize_predictions(
                    fold_targets, predictions[architecture])
                for architecture in predictions
            }
            fold_details.append(fold)
        if all_targets is None:
            all_targets = targets
        elif [(target.company, target.quarter) for target in targets] != [
                (target.company, target.quarter) for target in all_targets]:
            raise AssertionError("Runs produced a different test-quarter order")
        for architecture, values in fold_predictions.items():
            run_predictions[architecture].append(np.asarray(values, dtype=np.float64))
        run_details.append({"seed": run_seed, "folds": fold_details})

    ensemble_predictions = {
        architecture: np.mean(values, axis=0)
        for architecture, values in run_predictions.items()
    }
    metrics = {
        architecture: summarize_predictions(all_targets, prediction)
        for architecture, prediction in ensemble_predictions.items()
    }
    print("\n=== rolling future ensemble ===")
    for architecture, values in metrics.items():
        print("%-22s accuracy %.4f mcc %.4f mae %.4f"
              % (architecture, values["accuracy"], values["mcc"], values["mae"]))

    result = {
        "experiment": "past-only Top2Vec LSTM teacher plus quarter residual student",
        "device": str(device),
        "runs": args.runs,
        "seeds": [args.seed + run * 100000 for run in range(args.runs)],
        "evaluation": {
            "first_test_year": args.first_test_year,
            "last_test_year": args.last_test_year,
            "independent_unit": "company-quarter",
            "current_quarter_text": True,
            "target": "current-quarter financial percent_change four-class bucket",
            "top2vec_initialization": "frozen vectors trained unsupervised on full local corpus",
            "test_labels_used_for_training_or_selection": False,
        },
        "configuration": {
            "companies": company_names,
            "groups_per_quarter": args.groups_per_quarter,
            "group_size": {
                name: COMPANIES[name].getTweetGroupSize() for name in company_names},
            "max_tokens": args.max_tokens,
            "teacher_epochs": args.epochs,
            "teacher_patience": args.patience,
            "ridge_alphas": args.ridge_alphas,
            "text_summary_features": len(TEMPORAL_FEATURE_NAMES),
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
