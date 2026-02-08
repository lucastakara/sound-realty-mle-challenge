from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException

from app.api.schemas import (
    PredictionRequestFull,
    PredictionRequestMinimal,
    PredictionResponse,
)
from app.api.services import metadata, predict

router = APIRouter()


@router.get("/")
def root() -> dict:
    return {"message": "Sound Realty House Price Predictor API"}


@router.get("/health")
def health() -> dict:
    meta = metadata()
    return {"status": "ok", "model_version": meta["model_version"], "served_by": meta["served_by"]}


@router.post("/predict", response_model=PredictionResponse)
def predict_full(payload: PredictionRequestFull = Body(...)) -> PredictionResponse:
    """
    Performs housing price prediction using the FULL input schema.

    This endpoint expects a request body containing a `PredictionRequestFull`
    Pydantic model instance (aligned with the columns in
    `data/future_unseen_examples.csv`, excluding demographics fields).

    The service enriches the input on the backend by joining demographics from
    `data/zipcode_demographics.csv` using `zipcode`, then selects the ordered
    feature set from `app/model/model_features.json` (33 features) before calling
    `model.predict()`.

    Args:
        payload (PredictionRequestFull): Full set of non-demographic input features.

    Returns:
        PredictionResponse: JSON response containing:
            - prediction: model output
            - request_id: unique identifier for tracing/logging
            - latency_ms: end-to-end processing time in milliseconds
            - model_version, served_by, artifact paths: serving metadata

    Raises:
        HTTPException:
            - 422 (Validation Error): if request body does not match schema
            - 500 (Internal Server Error): if prediction pipeline fails unexpectedly
    """
    req_id = str(uuid4())
    try:
        y_pred, latency_ms = predict(payload.model_dump())
        meta = metadata()
        return PredictionResponse(
            prediction=y_pred,
            request_id=req_id,
            latency_ms=latency_ms,
            model_version=meta["model_version"],
            served_by=meta["served_by"],
            model_artifact_path=meta["model_artifact_path"],
            features_artifact_path=meta["features_artifact_path"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict_minimal", response_model=PredictionResponse)
def predict_minimal(payload: PredictionRequestMinimal = Body(...)) -> PredictionResponse:
    """
    Performs housing price prediction using the MINIMAL input schema (bonus endpoint).

    This endpoint expects a request body containing a `PredictionRequestMinimal`
    Pydantic model instance: only the minimal non-demographic house features plus
    `zipcode`.

    The service uses `zipcode` to enrich the request on the backend by joining
    demographics from `data/zipcode_demographics.csv`. It then selects the ordered
    feature set from `app/model/model_features.json` (33 features) before calling
    `model.predict()`.

    Args:
        payload (PredictionRequestMinimal): Minimal set of required input features
            plus `zipcode` for demographics enrichment.

    Returns:
        PredictionResponse: JSON response containing the prediction and useful
        serving metadata (request_id, latency, model version, instance, artifact paths).

    Raises:
        HTTPException:
            - 422 (Validation Error): if request body does not match schema
            - 500 (Internal Server Error): if prediction pipeline fails unexpectedly
    """
    req_id = str(uuid4())
    try:
        y_pred, latency_ms = predict(payload.model_dump())
        meta = metadata()
        return PredictionResponse(
            prediction=y_pred,
            request_id=req_id,
            latency_ms=latency_ms,
            model_version=meta["model_version"],
            served_by=meta["served_by"],
            model_artifact_path=meta["model_artifact_path"],
            features_artifact_path=meta["features_artifact_path"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))