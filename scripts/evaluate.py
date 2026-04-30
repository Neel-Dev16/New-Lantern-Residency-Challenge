#!/usr/bin/env python3
"""Evaluate the saved model on the case-level validation split."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import MODEL_PATH, THRESHOLD, VECTORIZER_PATH
from app.features import build_features, build_labeled_examples, has_left_right_mismatch

DATA_PATH = ROOT / "data" / "relevant_priors_public.json"
ERRORS_PATH = ROOT / "model_errors.txt"
RANDOM_STATE = 42


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def split_case_ids(examples: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    case_ids = sorted({example["case_id"] for example in examples})
    train_case_ids, validation_case_ids = train_test_split(
        case_ids,
        train_size=0.8,
        random_state=RANDOM_STATE,
    )
    train_case_id_set = set(train_case_ids)
    validation_case_id_set = set(validation_case_ids)
    overlap = train_case_id_set & validation_case_id_set
    if overlap:
        raise RuntimeError(f"case_id leakage across splits: {sorted(overlap)[:10]}")
    return train_case_id_set, validation_case_id_set


def examples_for_case_ids(
    examples: list[dict[str, Any]],
    case_ids: set[str],
) -> list[dict[str, Any]]:
    return [example for example in examples if example["case_id"] in case_ids]


def apply_prediction_rules(
    examples: list[dict[str, Any]],
    probabilities: Any,
) -> list[bool]:
    predictions: list[bool] = []
    for example, probability in zip(examples, probabilities, strict=True):
        if has_left_right_mismatch(
            example["current_description"],
            example["prior_description"],
        ):
            predictions.append(False)
        else:
            predictions.append(bool(probability >= THRESHOLD))
    return predictions


def sort_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    false_positives = [error for error in errors if error["error_type"] == "FP"]
    false_negatives = [error for error in errors if error["error_type"] == "FN"]
    false_positives.sort(key=lambda error: error["probability"], reverse=True)
    false_negatives.sort(key=lambda error: error["probability"])
    return false_positives + false_negatives


def write_errors(errors: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"Validation errors: {len(errors)}\n")
        file.write(f"Threshold: {THRESHOLD:.2f}\n")
        file.write("Split: case_id\n\n")

        for index, error in enumerate(errors, start=1):
            file.write(f"{index}.\n")
            file.write(f"  error type: {error['error_type']}\n")
            file.write(f"  case_id: {error['case_id']}\n")
            file.write(
                "  current study_description: "
                f"{error['current_description']}\n"
            )
            file.write(
                "  prior study_description: "
                f"{error['prior_description']}\n"
            )
            file.write(f"  current study_date: {error['current_date']}\n")
            file.write(f"  prior study_date: {error['prior_date']}\n")
            file.write(
                "  predicted probability: "
                f"{error['probability']:.6f}\n"
            )
            file.write(f"  true label: {error['true_label']}\n\n")


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    examples = build_labeled_examples(dataset)
    train_case_ids, validation_case_ids = split_case_ids(examples)
    validation_examples = examples_for_case_ids(examples, validation_case_ids)
    validation_labels = [example["label"] for example in validation_examples]

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    validation_features = build_features(vectorizer, validation_examples)

    probabilities = model.predict_proba(validation_features)[:, 1]
    predictions = apply_prediction_rules(validation_examples, probabilities)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        validation_labels,
        predictions,
        labels=[False, True],
    ).ravel()

    errors: list[dict[str, Any]] = []
    for example, probability, prediction, true_label in zip(
        validation_examples,
        probabilities,
        predictions,
        validation_labels,
        strict=True,
    ):
        if bool(prediction) == bool(true_label):
            continue

        errors.append(
            {
                "error_type": "FP" if prediction else "FN",
                "case_id": example["case_id"],
                "current_description": example["current_description"],
                "prior_description": example["prior_description"],
                "current_date": example["current_date"],
                "prior_date": example["prior_date"],
                "probability": float(probability),
                "true_label": bool(true_label),
            }
        )

    sorted_errors = sort_errors(errors)
    write_errors(sorted_errors, ERRORS_PATH)

    accuracy = accuracy_score(validation_labels, predictions)
    precision = precision_score(validation_labels, predictions, zero_division=0)
    recall = recall_score(validation_labels, predictions, zero_division=0)

    print(f"Train cases: {len(train_case_ids)}")
    print(f"Validation cases: {len(validation_case_ids)}")
    print(f"Total predictions: {len(validation_labels)}")
    print(f"Correct predictions: {len(validation_labels) - len(errors)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"False positives: {int(false_positive)}")
    print(f"False negatives: {int(false_negative)}")
    print(f"Wrote validation errors: {ERRORS_PATH}")


if __name__ == "__main__":
    main()
