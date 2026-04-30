"""FastAPI prediction service for relevant prior studies."""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from app.config import MODEL_PATH, THRESHOLD, VECTORIZER_PATH
from app.features import build_features, has_left_right_mismatch


app = FastAPI(title="New Lantern Priors Predictor")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
logger = logging.getLogger("uvicorn.error")


class Study(BaseModel):
    model_config = ConfigDict(extra="ignore")

    study_id: str
    study_description: str
    study_date: str


class Case(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    current_study: Study
    prior_studies: list[Study]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cases: list[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]


def collect_examples(cases: list[Case]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for case in cases:
        current_description = case.current_study.study_description

        for prior_study in case.prior_studies:
            examples.append(
                {
                    "case_id": case.case_id,
                    "study_id": prior_study.study_id,
                    "current_description": current_description,
                    "prior_description": prior_study.study_description,
                }
            )

    return examples


def predict_batch(examples: list[dict[str, Any]]) -> list[Prediction]:
    if not examples:
        return []

    features = build_features(vectorizer, examples)
    probabilities = model.predict_proba(features)[:, 1]

    predictions = []
    for example, probability in zip(examples, probabilities, strict=True):
        if has_left_right_mismatch(
            example["current_description"],
            example["prior_description"],
        ):
            predicted_is_relevant = False
        else:
            predicted_is_relevant = bool(probability >= THRESHOLD)

        predictions.append(
            Prediction(
                case_id=example["case_id"],
                study_id=example["study_id"],
                predicted_is_relevant=predicted_is_relevant,
            )
        )

    return predictions


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start_time = time.perf_counter()
    examples = collect_examples(payload.cases)
    predictions = predict_batch(examples)
    elapsed_seconds = time.perf_counter() - start_time

    logger.info(
        "prediction_request cases=%d prior_studies=%d elapsed_seconds=%.4f",
        len(payload.cases),
        len(examples),
        elapsed_seconds,
    )

    return PredictResponse(predictions=predictions)
