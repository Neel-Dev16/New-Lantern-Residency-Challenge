#!/usr/bin/env python3
"""Train a TF-IDF + LogisticRegression relevance model."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "relevant_priors_public.json"
MODEL_PATH = ROOT / "models" / "model.joblib"
VECTORIZER_PATH = ROOT / "models" / "vectorizer.joblib"
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


def build_training_data(
    dataset: dict[str, Any],
) -> tuple[list[str], list[str], list[bool]]:
    truth_map = build_truth_map(dataset)
    current_descriptions: list[str] = []
    prior_descriptions: list[str] = []
    labels: list[bool] = []

    for case in dataset["cases"]:
        case_id = str(case["case_id"])
        current_description = str(
            case["current_study"].get("study_description", "")
        )

        for prior_study in case["prior_studies"]:
            study_id = str(prior_study["study_id"])
            prior_description = str(prior_study.get("study_description", ""))
            current_descriptions.append(current_description)
            prior_descriptions.append(prior_description)
            labels.append(truth_map[(case_id, study_id)])

    return current_descriptions, prior_descriptions, labels


def build_numeric_and_rule_features(
    current_descriptions: list[str],
    prior_descriptions: list[str],
    current_vectors: csr_matrix,
    prior_vectors: csr_matrix,
) -> csr_matrix:
    cosine_similarities = np.asarray(
        current_vectors.multiply(prior_vectors).sum(axis=1)
    ).ravel()
    numeric_features = np.zeros((len(current_descriptions), 5), dtype=float)

    for index, (current_description, prior_description) in enumerate(
        zip(current_descriptions, prior_descriptions, strict=True)
    ):
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
    vectorizer: TfidfVectorizer,
    current_descriptions: list[str],
    prior_descriptions: list[str],
) -> csr_matrix:
    current_vectors = vectorizer.transform(current_descriptions)
    prior_vectors = vectorizer.transform(prior_descriptions)
    numeric_and_rule_features = build_numeric_and_rule_features(
        current_descriptions,
        prior_descriptions,
        current_vectors,
        prior_vectors,
    )
    return hstack(
        [current_vectors, prior_vectors, numeric_and_rule_features],
        format="csr",
    )


def evaluate_thresholds(
    validation_labels: list[bool],
    probabilities: np.ndarray,
) -> None:
    best_threshold = 0.0
    best_accuracy = -1.0

    print("Threshold metrics")
    print("-----------------")
    for threshold_index in range(10, 91):
        threshold = threshold_index / 100
        predictions = probabilities >= threshold

        accuracy = accuracy_score(validation_labels, predictions)
        precision = precision_score(validation_labels, predictions, zero_division=0)
        recall = recall_score(validation_labels, predictions, zero_division=0)
        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(validation_labels, predictions, labels=[False, True])
            .ravel()
            .tolist()
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

        print(
            f"threshold={threshold:.2f} "
            f"accuracy={accuracy:.4f} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"false_positives={false_positive} "
            f"false_negatives={false_negative}"
        )

    print("\nBest threshold by validation accuracy")
    print("-------------------------------------")
    print(f"threshold={best_threshold:.2f} accuracy={best_accuracy:.4f}")


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    current_descriptions, prior_descriptions, labels = build_training_data(dataset)

    (
        train_current_descriptions,
        validation_current_descriptions,
        train_prior_descriptions,
        validation_prior_descriptions,
        train_labels,
        validation_labels,
    ) = train_test_split(
        current_descriptions,
        prior_descriptions,
        labels,
        train_size=0.8,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)
    vectorizer.fit(train_current_descriptions + train_prior_descriptions)
    train_features = build_features(
        vectorizer,
        train_current_descriptions,
        train_prior_descriptions,
    )
    validation_features = build_features(
        vectorizer,
        validation_current_descriptions,
        validation_prior_descriptions,
    )

    model = LogisticRegression(max_iter=1_000)
    model.fit(train_features, train_labels)

    probabilities = model.predict_proba(validation_features)[:, 1]
    evaluate_thresholds(validation_labels, probabilities)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
