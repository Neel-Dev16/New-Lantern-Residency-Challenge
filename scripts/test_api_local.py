#!/usr/bin/env python3
"""Smoke test the local FastAPI /predict endpoint."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "relevant_priors_public.json"
PREDICT_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
NUM_CASES = 3


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_sample_payload(dataset: dict[str, Any], num_cases: int) -> dict[str, Any]:
    cases = dataset["cases"][:num_cases]
    return {"cases": cases}


def count_prior_studies(payload: dict[str, Any]) -> int:
    return sum(len(case["prior_studies"]) for case in payload["cases"])


def validate_response(
    response_json: dict[str, Any],
    expected_prediction_count: int,
) -> None:
    if "predictions" not in response_json:
        raise AssertionError('Response is missing "predictions"')

    predictions = response_json["predictions"]
    if not isinstance(predictions, list):
        raise AssertionError('"predictions" must be a list')

    if len(predictions) != expected_prediction_count:
        raise AssertionError(
            "Prediction count mismatch: "
            f"expected {expected_prediction_count}, got {len(predictions)}"
        )

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


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    payload = build_sample_payload(dataset, NUM_CASES)
    prior_count = count_prior_studies(payload)

    start_time = time.perf_counter()
    response = requests.post(PREDICT_URL, json=payload, timeout=30)
    request_time = time.perf_counter() - start_time

    if response.status_code != 200:
        raise AssertionError(
            f"Expected status code 200, got {response.status_code}: {response.text}"
        )

    validate_response(response.json(), prior_count)

    print(
        "Local API test passed: "
        f"cases={len(payload['cases'])}, "
        f"prior_studies={prior_count}, "
        f"request_time={request_time:.4f}s"
    )


if __name__ == "__main__":
    main()
