from __future__ import annotations

import json
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# IMPORTANT: these paths are *inside the container*
# We will bake artifacts into /app/app/model and /app/app/data
MODEL_DIR = Path("app/model")
DATA_DIR = Path("app/data")

MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.json"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


MODEL_VERSION = _env("MODEL_VERSION", "baseline-v1")
SERVED_BY = _env("SERVED_BY", "unknown")  # "blue" or "green"


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_features() -> List[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list not found at {FEATURES_PATH}.")
    with open(FEATURES_PATH, "r") as f:
        feats = json.load(f)

    if not isinstance(feats, list) or not feats:
        raise ValueError("model_features.json is not a non-empty list.")
    return feats


@lru_cache(maxsize=1)
def load_demographics() -> pd.DataFrame:
    if not DEMOGRAPHICS_PATH.exists():
        raise FileNotFoundError(f"Demographics not found at {DEMOGRAPHICS_PATH}.")
    return pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})


def enrich_with_demographics(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Requirement: input must NOT include demographics.
    We join demographics on the backend using zipcode.
    """
    demographics = load_demographics()
    df = df_input.copy()
    df["zipcode"] = df["zipcode"].astype(str)

    merged = df.merge(demographics, how="left", on="zipcode")

    # baseline training dropped zipcode after join
    if "zipcode" in merged.columns:
        merged = merged.drop(columns=["zipcode"])

    return merged


def predict(payload: Dict[str, Any]) -> Tuple[float, float]:
    """
    Returns:
      - prediction
      - latency_ms
    """
    start = time.perf_counter()

    model = load_model()
    features = load_features()

    df_input = pd.DataFrame([payload])
    df_full = enrich_with_demographics(df_input)

    missing = [c for c in features if c not in df_full.columns]
    if missing:
        raise ValueError(f"Missing required feature columns after enrichment: {missing}")

    X = df_full[features]  # enforce correct column order
    y_pred = float(model.predict(X)[0])

    latency_ms = (time.perf_counter() - start) * 1000.0
    return y_pred, latency_ms


def metadata() -> Dict[str, str]:
    return {
        "model_version": MODEL_VERSION,
        "served_by": SERVED_BY,
        "model_artifact_path": str(MODEL_PATH),
        "features_artifact_path": str(FEATURES_PATH),
    }