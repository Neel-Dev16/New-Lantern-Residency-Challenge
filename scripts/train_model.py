#!/usr/bin/env python3
"""Train a TF-IDF + LogisticRegression relevance model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import MODEL_PATH, THRESHOLD, VECTORIZER_PATH
from app.features import build_features, build_labeled_examples, has_left_right_mismatch

DATA_PATH = ROOT / "data" / "relevant_priors_public.json"
METADATA_PATH = ROOT / "models" / "metadata.json"
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


def descriptions_for_vectorizer(examples: list[dict[str, Any]]) -> list[str]:
    descriptions: list[str] = []
    for example in examples:
        descriptions.append(example["current_description"])
        descriptions.append(example["prior_description"])
    return descriptions


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


def write_metadata(
    train_case_ids: set[str],
    validation_case_ids: set[str],
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    metrics: dict[str, float | int],
) -> None:
    metadata = {
        "threshold": THRESHOLD,
        "split": {
            "type": "case_id",
            "random_state": RANDOM_STATE,
            "train_cases": len(train_case_ids),
            "validation_cases": len(validation_case_ids),
            "train_rows": len(train_examples),
            "validation_rows": len(validation_examples),
        },
        "validation_metrics": metrics,
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    examples = build_labeled_examples(dataset)
    train_case_ids, validation_case_ids = split_case_ids(examples)
    train_examples = examples_for_case_ids(examples, train_case_ids)
    validation_examples = examples_for_case_ids(examples, validation_case_ids)

    train_labels = [example["label"] for example in train_examples]
    validation_labels = [example["label"] for example in validation_examples]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)
    vectorizer.fit(descriptions_for_vectorizer(train_examples))

    train_features = build_features(vectorizer, train_examples)
    validation_features = build_features(vectorizer, validation_examples)

    model = LogisticRegression(max_iter=1_000)
    model.fit(train_features, train_labels)

    probabilities = model.predict_proba(validation_features)[:, 1]
    predictions = apply_prediction_rules(validation_examples, probabilities)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        validation_labels,
        predictions,
        labels=[False, True],
    ).ravel()

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "precision": float(
            precision_score(validation_labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(validation_labels, predictions, zero_division=0)),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
        "true_negatives": int(true_negative),
    }

    print(f"Train cases: {len(train_case_ids)}")
    print(f"Validation cases: {len(validation_case_ids)}")
    print(f"Train rows: {len(train_examples)}")
    print(f"Validation rows: {len(validation_examples)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    write_metadata(
        train_case_ids,
        validation_case_ids,
        train_examples,
        validation_examples,
        metrics,
    )

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
