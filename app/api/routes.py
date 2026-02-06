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