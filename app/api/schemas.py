from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequestFull(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., ge=0)
    sqft_lot: int = Field(..., ge=0)
    floors: float = Field(..., ge=0)
    waterfront: int = Field(..., ge=0)
    view: int = Field(..., ge=0)
    condition: int = Field(..., ge=0)
    grade: int = Field(..., ge=0)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    yr_built: int = Field(..., ge=0)
    yr_renovated: int = Field(..., ge=0)
    zipcode: str
    lat: float
    long: float
    sqft_living15: int = Field(..., ge=0)
    sqft_lot15: int = Field(..., ge=0)


class PredictionRequestMinimal(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., ge=0)
    sqft_lot: int = Field(..., ge=0)
    floors: float = Field(..., ge=0)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    zipcode: str


class PredictionResponse(BaseModel):
    # avoid pydantic warning for "model_*"
    model_config = ConfigDict(protected_namespaces=())

    prediction: float
    request_id: str
    latency_ms: float = Field(..., ge=0)


    model_version: str
    served_by: str

    # optional “useful metadata”
    model_artifact_path: str
    features_artifact_path: str