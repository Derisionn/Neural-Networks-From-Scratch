"""
main.py
-------
FastAPI application that exposes the trained neural network as a REST API.

Endpoints:
    GET  /              → Welcome message
    GET  /health        → API health check
    POST /predict       → Single-sample prediction
    POST /predict_batch → Batch prediction

Run with:
    uvicorn main:app --reload
"""

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List

from model import NeuralNetwork


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for a single prediction."""
    features: List[float]

    @field_validator("features")
    @classmethod
    def must_have_two_features(cls, v):
        if len(v) != 2:
            raise ValueError(f"Expected exactly 2 features, got {len(v)}.")
        return v


class PredictResponse(BaseModel):
    """Response body for a single prediction."""
    predicted_class: int
    probabilities: List[float]


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""
    samples: List[List[float]]

    @field_validator("samples")
    @classmethod
    def each_sample_must_have_two_features(cls, v):
        for i, sample in enumerate(v):
            if len(sample) != 2:
                raise ValueError(
                    f"Sample at index {i} has {len(sample)} features; expected 2."
                )
        return v


class BatchPredictResponse(BaseModel):
    """Response body for batch prediction."""
    predictions: List[int]
    probabilities: List[List[float]]


# ---------------------------------------------------------------------------
# Application lifecycle  (load model once at startup)
# ---------------------------------------------------------------------------

# Global model instance — populated during startup
nn: NeuralNetwork | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights when the server starts; clean up on shutdown."""
    global nn
    try:
        # Weights are expected in the same directory as main.py
        nn = NeuralNetwork(weights_dir=".")
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError as e:
        print(f"⚠️  WARNING: {e}")
        print("   The /predict and /predict_batch endpoints will return 503 until weights are present.")
    yield
    # Cleanup (nothing to do here, but the hook is available)
    nn = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Neural Network API",
    description=(
        "A production-style REST API that serves a two-layer neural network "
        "trained from scratch with NumPy on the spiral dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_model() -> NeuralNetwork:
    """Raise 503 if the model was not loaded (missing weight files)."""
    if nn is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model weights are not loaded. "
                "Please add the .npy weight files and restart the server."
            ),
        )
    return nn


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root():
    """Welcome endpoint."""
    return {
        "message": "Welcome to the Neural Network API 🧠",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
def health():
    """Health-check endpoint — indicates whether the model is ready."""
    return {
        "status": "ok",
        "model_loaded": nn is not None,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Predict the class for a single input sample.

    **Request body:**
    ```json
    { "features": [x1, x2] }
    ```

    **Response:**
    ```json
    {
        "predicted_class": 0,
        "probabilities": [0.91, 0.06, 0.03]
    }
    ```
    """
    model = _require_model()

    X = np.array(request.features, dtype=np.float32)
    probs = model.forward(X)          # shape (3,)
    predicted_class = int(np.argmax(probs))

    return PredictResponse(
        predicted_class=predicted_class,
        probabilities=[round(float(p), 6) for p in probs],
    )


@app.post("/predict_batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest):
    """
    Predict classes for a batch of input samples.

    **Request body:**
    ```json
    { "samples": [[x1, x2], [x3, x4], ...] }
    ```

    **Response:**
    ```json
    {
        "predictions": [0, 2, 1],
        "probabilities": [[0.91, 0.06, 0.03], ...]
    }
    ```
    """
    model = _require_model()

    X = np.array(request.samples, dtype=np.float32)   # shape (n, 2)
    probs = model.forward(X)                           # shape (n, 3)
    predictions = np.argmax(probs, axis=-1).tolist()

    return BatchPredictResponse(
        predictions=predictions,
        probabilities=[[round(float(p), 6) for p in row] for row in probs],
    )
