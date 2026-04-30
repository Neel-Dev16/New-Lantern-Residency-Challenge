"""Shared preprocessing and feature generation."""

from __future__ import annotations

import string
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack


PUNCTUATION_TRANSLATION = str.maketrans(
    {character: " " for character in string.punctuation}
)


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


def build_truth_map(dataset: dict[str, Any]) -> dict[tuple[str, str], bool]:
    truth_map: dict[tuple[str, str], bool] = {}
    for label in dataset["truth"]:
        key = (str(label["case_id"]), str(label["study_id"]))
        truth_map[key] = bool(label["is_relevant_to_current"])
    return truth_map


def build_labeled_examples(dataset: dict[str, Any]) -> list[dict[str, Any]]:
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


def build_features(vectorizer: Any, examples: list[dict[str, Any]]) -> csr_matrix:
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
