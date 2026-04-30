#!/usr/bin/env python3
"""Validate /predict against the full public dataset payload."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "relevant_priors_public.json"
PREDICT_URL = "http://127.0.0.1:8000/predict"


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def expected_prediction_keys(cases: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for case in cases:
        case_id = str(case["case_id"])
        for prior_study in case["prior_studies"]:
            keys.add((case_id, str(prior_study["study_id"])))
    return keys


def count_prior_studies(cases: list[dict[str, Any]]) -> int:
    return sum(len(case["prior_studies"]) for case in cases)


def validate_predictions(
    response_json: dict[str, Any],
    expected_keys: set[tuple[str, str]],
    expected_count: int,
) -> list[dict[str, Any]]:
    if "predictions" not in response_json:
        raise AssertionError('Response is missing "predictions"')

    predictions = response_json["predictions"]
    if not isinstance(predictions, list):
        raise AssertionError('"predictions" must be a list')

    if len(predictions) != expected_count:
        raise AssertionError(
            "Prediction count mismatch: "
            f"expected {expected_count}, got {len(predictions)}"
        )

    seen_keys: set[tuple[str, str]] = set()
    duplicate_keys: set[tuple[str, str]] = set()
    required_fields = {"case_id", "study_id", "predicted_is_relevant"}

    for index, prediction in enumerate(predictions):
        missing_fields = required_fields - set(prediction)
        if missing_fields:
            raise AssertionError(
                f"Prediction {index} missing fields: {sorted(missing_fields)}"
            )

        if not isinstance(prediction["predicted_is_relevant"], bool):
            raise AssertionError(
                f"Prediction {index} predicted_is_relevant must be boolean"
            )

        key = (str(prediction["case_id"]), str(prediction["study_id"]))
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)

    if duplicate_keys:
        preview = sorted(duplicate_keys)[:10]
        raise AssertionError(f"Duplicate prediction keys found: {preview}")

    missing_keys = expected_keys - seen_keys
    if missing_keys:
        preview = sorted(missing_keys)[:10]
        raise AssertionError(f"Missing prediction keys found: {preview}")

    unexpected_keys = seen_keys - expected_keys
    if unexpected_keys:
        preview = sorted(unexpected_keys)[:10]
        raise AssertionError(f"Unexpected prediction keys found: {preview}")

    return predictions


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    cases = dataset["cases"]
    payload = {"cases": cases}
    prior_count = count_prior_studies(cases)
    expected_keys = expected_prediction_keys(cases)

    start_time = time.perf_counter()
    response = requests.post(PREDICT_URL, json=payload, timeout=120)
    request_time = time.perf_counter() - start_time

    if response.status_code != 200:
        raise AssertionError(
            f"Expected status code 200, got {response.status_code}: {response.text}"
        )

    predictions = validate_predictions(response.json(), expected_keys, prior_count)

    print(f"Cases: {len(cases)}")
    print(f"Prior studies: {prior_count}")
    print(f"Predictions returned: {len(predictions)}")
    print(f"Request time: {request_time:.4f}s")


if __name__ == "__main__":
    main()
