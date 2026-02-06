# app/api/main.py
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as inference_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sound Realty - Housing Price API",
        description="REST API for house price prediction (baseline model).",
        version="0.1.0",
    )

    # Routes (/ , /health, /predict, /predict_minimal)
    app.include_router(inference_router)

    return app


app = create_app()