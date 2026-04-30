#!/usr/bin/env python3
"""Evaluate a baseline relevance predictor on the public relevant-priors data."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "relevant_priors_public.json"
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
    """Lowercase, remove light punctuation differences, and collapse spaces."""
    without_punctuation = description.lower().translate(PUNCTUATION_TRANSLATION)
    return " ".join(without_punctuation.split())


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


def contains_spine_term(description: str) -> bool:
    normalized = normalize_description(description)
    return "spine" in normalized or "thoracic spine" in normalized


def predict(case: dict[str, Any], prior_study: dict[str, Any]) -> bool:
    """Predict relevance using exact description match, then body-region rules."""
    current_description = case["current_study"].get("study_description", "")
    prior_description = prior_study.get("study_description", "")

    if normalize_description(current_description) == normalize_description(
        prior_description
    ):
        return True

    current_region = extract_body_region(current_description)
    prior_region = extract_body_region(prior_description)

    if current_region == "unknown" or prior_region == "unknown":
        return False

    if current_region == prior_region == "breast":
        return True
    if current_region == prior_region == "chest" and not contains_spine_term(
        prior_description
    ):
        return True
    if current_region == prior_region == "spine":
        return True
    if current_region == prior_region and current_region in {
        "abdomen_pelvis",
        "kidney_urinary",
        "brain_head",
        "extremity",
        "cardiac",
    }:
        return True
    if current_region == "kidney_urinary" and prior_region == "abdomen_pelvis":
        return True
    if current_region == "cardiac" and prior_region == "chest":
        return True

    return False


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    truth_map = build_truth_map(dataset)

    total_predictions = 0
    correct_predictions = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    false_negative_examples: list[dict[str, str]] = []

    for case in dataset["cases"]:
        case_id = str(case["case_id"])
        for prior_study in case["prior_studies"]:
            study_id = str(prior_study["study_id"])
            truth = truth_map[(case_id, study_id)]
            prediction = predict(case, prior_study)

            total_predictions += 1
            if prediction and truth:
                true_positives += 1
            elif not prediction and not truth:
                true_negatives += 1
            elif prediction and not truth:
                false_positives += 1
            elif not prediction and truth:
                false_negatives += 1
                if len(false_negative_examples) < 100:
                    false_negative_examples.append(
                        {
                            "case_id": case_id,
                            "current_description": str(
                                case["current_study"].get("study_description", "")
                            ),
                            "prior_description": str(
                                prior_study.get("study_description", "")
                            ),
                            "current_date": str(
                                case["current_study"].get("study_date", "")
                            ),
                            "prior_date": str(prior_study.get("study_date", "")),
                        }
                    )

    correct_predictions = true_positives + true_negatives
    accuracy = correct_predictions / total_predictions if total_predictions else 0.0
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"True positives: {true_positives}")
    print(f"True negatives: {true_negatives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nFalse negative examples")
    print("-----------------------")
    for index, example in enumerate(false_negative_examples, start=1):
        print(f"{index}.")
        print(f"  case_id: {example['case_id']}")
        print(f"  current study_description: {example['current_description']}")
        print(f"  prior study_description: {example['prior_description']}")
        print(f"  current study_date: {example['current_date']}")
        print(f"  prior study_date: {example['prior_date']}")


if __name__ == "__main__":
    main()
