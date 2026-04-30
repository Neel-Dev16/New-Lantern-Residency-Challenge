#!/usr/bin/env python3
"""Evaluate the saved pure ML model on the validation split."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "relevant_priors_public.json"
MODEL_PATH = ROOT / "models" / "model.joblib"
VECTORIZER_PATH = ROOT / "models" / "vectorizer.joblib"
ERRORS_PATH = ROOT / "model_errors.txt"
BEST_THRESHOLD = 0.51
PUNCTUATION_TRANSLATION = str.maketrans(
    {character: " " for character in string.punctuation}
)


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_truth_map(dataset: dict[str, Any]) -> dict[tuple[str, str], bool]:
    truth_map: dict[tuple[str, str], bool] = {}
    for label in dataset["truth"]:
        key = (str(label["case_id"]), str(label["study_id"]))
        truth_map[key] = bool(label["is_relevant_to_current"])
    return truth_map


def normalize_description(description: str) -> str:
    without_punctuation = description.lower().translate(PUNCTUATION_TRANSLATION)
    return " ".join(without_punctuation.split())


def tokenize(description: str) -> set[str]:
    return set(normalize_description(description).split())


def extract_body_region(description: str) -> str:
    normalized = normalize_description(description)

    region_keywords = {
        "breast": (
            "mam",
            "mammo",
            "mammography",
            "breast",
            "tomo",
            "ultrasound breast",
            "mri breast",
        ),
        "spine": ("cervical", "thoracic spine", "lumbar", "spine"),
        "chest": ("chest", "thorax", "rib", "ribs", "lung"),
        "cardiac": (
            "echo",
            "coronary",
            "cardiac",
            "myo perf",
            "myocardial",
            "spect",
            "stress",
        ),
        "abdomen_pelvis": ("abdomen", "abdominal", "pelvis", "abd", "plv"),
        "kidney_urinary": (
            "kidney",
            "kidneys",
            "bladder",
            "renal",
            "urogram",
            "urinary",
        ),
        "brain_head": ("brain", "head", "stroke"),
        "extremity": (
            "knee",
            "ankle",
            "foot",
            "hand",
            "wrist",
            "shoulder",
            "hip",
            "femur",
            "tibia",
            "humerus",
            "elbow",
        ),
    }

    for region, keywords in region_keywords.items():
        if any(keyword in normalized for keyword in keywords):
            return region
    return "unknown"


def has_left_right_mismatch(
    current_description: str,
    prior_description: str,
) -> bool:
    current_tokens = tokenize(current_description)
    prior_tokens = tokenize(prior_description)
    current_has_left = "left" in current_tokens or "lt" in current_tokens
    current_has_right = "right" in current_tokens or "rt" in current_tokens
    prior_has_left = "left" in prior_tokens or "lt" in prior_tokens
    prior_has_right = "right" in prior_tokens or "rt" in prior_tokens

    return (
        (current_has_left and prior_has_right)
        or (current_has_right and prior_has_left)
    )


def build_examples(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    truth_map = build_truth_map(dataset)
    examples: list[dict[str, Any]] = []

    for case in dataset["cases"]:
        case_id = str(case["case_id"])
        current_study = case["current_study"]
        current_description = str(current_study.get("study_description", ""))
        current_date = str(current_study.get("study_date", ""))

        for prior_study in case["prior_studies"]:
            study_id = str(prior_study["study_id"])
            examples.append(
                {
                    "case_id": case_id,
                    "study_id": study_id,
                    "current_description": current_description,
                    "prior_description": str(
                        prior_study.get("study_description", "")
                    ),
                    "current_date": current_date,
                    "prior_date": str(prior_study.get("study_date", "")),
                    "label": truth_map[(case_id, study_id)],
                }
            )

    return examples


def build_numeric_and_rule_features(
    examples: list[dict[str, Any]],
    current_vectors: csr_matrix,
    prior_vectors: csr_matrix,
) -> csr_matrix:
    cosine_similarities = np.asarray(
        current_vectors.multiply(prior_vectors).sum(axis=1)
    ).ravel()
    numeric_features = np.zeros((len(examples), 5), dtype=float)

    for index, example in enumerate(examples):
        current_description = example["current_description"]
        prior_description = example["prior_description"]
        current_tokens = tokenize(current_description)
        prior_tokens = tokenize(prior_description)
        overlap_count = len(current_tokens & prior_tokens)
        overlap_ratio = (
            overlap_count / len(current_tokens | prior_tokens)
            if current_tokens or prior_tokens
            else 0.0
        )
        same_exact_description = int(
            normalize_description(current_description)
            == normalize_description(prior_description)
        )
        same_body_region = int(
            extract_body_region(current_description) != "unknown"
            and extract_body_region(current_description)
            == extract_body_region(prior_description)
        )

        numeric_features[index] = [
            cosine_similarities[index],
            overlap_count,
            overlap_ratio,
            same_exact_description,
            same_body_region,
        ]

    return csr_matrix(numeric_features)


def build_features(
    vectorizer: Any,
    examples: list[dict[str, Any]],
) -> csr_matrix:
    current_descriptions = [example["current_description"] for example in examples]
    prior_descriptions = [example["prior_description"] for example in examples]
    current_vectors = vectorizer.transform(current_descriptions)
    prior_vectors = vectorizer.transform(prior_descriptions)
    numeric_and_rule_features = build_numeric_and_rule_features(
        examples,
        current_vectors,
        prior_vectors,
    )
    return hstack(
        [current_vectors, prior_vectors, numeric_and_rule_features],
        format="csr",
    )


def sort_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    false_positives = [
        error for error in errors if error["error_type"] == "FP"
    ]
    false_negatives = [
        error for error in errors if error["error_type"] == "FN"
    ]
    false_positives.sort(key=lambda error: error["probability"], reverse=True)
    false_negatives.sort(key=lambda error: error["probability"])
    return false_positives + false_negatives


def write_errors(errors: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"Validation errors: {len(errors)}\n")
        file.write(f"Threshold: {BEST_THRESHOLD:.2f}\n\n")

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
    examples = build_examples(dataset)
    labels = [example["label"] for example in examples]

    _, validation_examples = train_test_split(
        examples,
        train_size=0.8,
        random_state=42,
        stratify=labels,
    )
    validation_labels = [example["label"] for example in validation_examples]

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    validation_features = build_features(vectorizer, validation_examples)

    probabilities = model.predict_proba(validation_features)[:, 1]
    ml_predictions = probabilities >= BEST_THRESHOLD
    predictions = np.array(
        [
            False if has_left_right_mismatch(
                example["current_description"],
                example["prior_description"],
            ) else bool(ml_prediction)
            for example, ml_prediction in zip(
                validation_examples,
                ml_predictions,
                strict=True,
            )
        ],
        dtype=bool,
    )

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
    false_positives = sum(error["error_type"] == "FP" for error in errors)
    false_negatives = sum(error["error_type"] == "FN" for error in errors)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total predictions: {len(validation_labels)}")
    print(f"Correct predictions: {len(validation_labels) - len(errors)}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Wrote validation errors: {ERRORS_PATH}")


if __name__ == "__main__":
    main()
