#!/usr/bin/env python3
"""Summarize the public relevant-priors dataset.

This script only inspects the dataset. It does not train or save a model.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "relevant_priors_public.json"


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def print_top_counts(title: str, counts: Counter[str], limit: int = 30) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for rank, (description, count) in enumerate(counts.most_common(limit), start=1):
        print(f"{rank:2d}. {count:5d}  {description}")


def main() -> None:
    dataset = load_dataset(DATA_PATH)

    cases = dataset.get("cases", [])
    truth = dataset.get("truth", [])

    truth_by_key: dict[tuple[str, str], bool] = {}
    duplicate_truth_keys: list[tuple[str, str]] = []
    relevant_count = 0
    not_relevant_count = 0

    for label in truth:
        key = (str(label["case_id"]), str(label["study_id"]))
        if key in truth_by_key:
            duplicate_truth_keys.append(key)
        truth_by_key[key] = bool(label["is_relevant_to_current"])
        if label["is_relevant_to_current"]:
            relevant_count += 1
        else:
            not_relevant_count += 1

    current_description_counts: Counter[str] = Counter()
    prior_description_counts: Counter[str] = Counter()
    prior_count = 0
    missing_truth_keys: list[tuple[str, str]] = []

    for case in cases:
        case_id = str(case["case_id"])

        current_study = case.get("current_study", {})
        current_description = current_study.get("study_description")
        if current_description:
            current_description_counts[str(current_description)] += 1

        for prior_study in case.get("prior_studies", []):
            prior_count += 1

            prior_description = prior_study.get("study_description")
            if prior_description:
                prior_description_counts[str(prior_description)] += 1

            key = (case_id, str(prior_study["study_id"]))
            if key not in truth_by_key:
                missing_truth_keys.append(key)

    average_priors = prior_count / len(cases) if cases else 0.0

    print(f"Dataset: {DATA_PATH}")
    print(f"Number of cases: {len(cases)}")
    print(f"Number of truth labels: {len(truth)}")
    print(f"Relevant labels: {relevant_count}")
    print(f"Not relevant labels: {not_relevant_count}")
    print(f"Average prior studies per case: {average_priors:.2f}")

    print_top_counts("Top 30 current study descriptions", current_description_counts)
    print_top_counts("Top 30 prior study descriptions", prior_description_counts)

    print("\nTruth label verification")
    print("------------------------")
    print(f"Prior studies checked: {prior_count}")
    print(f"Missing truth labels for prior studies: {len(missing_truth_keys)}")
    print(f"Duplicate truth label keys: {len(duplicate_truth_keys)}")

    if missing_truth_keys:
        print("\nFirst missing truth label keys:")
        for case_id, study_id in missing_truth_keys[:30]:
            print(f"- case_id={case_id}, study_id={study_id}")
    else:
        print("Every prior study has a matching truth label by case_id + study_id.")


if __name__ == "__main__":
    main()
