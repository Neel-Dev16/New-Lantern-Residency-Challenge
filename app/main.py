"""FastAPI prediction service for relevant prior studies."""

from __future__ import annotations

import string
import time
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from scipy.sparse import csr_matrix, hstack


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
VECTORIZER_PATH = ROOT / "models" / "vectorizer.joblib"
BEST_THRESHOLD = 0.51
PUNCTUATION_TRANSLATION = str.maketrans(
    {character: " " for character in string.punctuation}
)


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


def generate_features(examples: list[dict[str, Any]]) -> csr_matrix:
    """Generate the same batch feature matrix used during training."""
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

    features = generate_features(examples)
    probabilities = model.predict_proba(features)[:, 1]

    predictions = []
    for example, probability in zip(examples, probabilities, strict=True):
        if has_left_right_mismatch(
            example["current_description"],
            example["prior_description"],
        ):
            predicted_is_relevant = False
        else:
            predicted_is_relevant = bool(probability >= BEST_THRESHOLD)

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
