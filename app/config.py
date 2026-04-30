"""Shared configuration for model inference."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
VECTORIZER_PATH = ROOT / "models" / "vectorizer.joblib"
THRESHOLD = 0.51
