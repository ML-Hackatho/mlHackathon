"""FastAPI app for model-backed PathSnap trip prediction.

Configure PATHSNAP_TRIPS_DIR, PATHSNAP_MODEL_FILE and PATHSNAP_DECODER_CONFIG
before starting the server. Prediction never reads a trip's .route file; when a
reference route exists, aggregate-only evaluation is computed after prediction.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import RoutePredictor, score_against_reference

LOG = logging.getLogger("pathsnap-person2.api")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(env_name: str, default: str) -> Path:
    raw = Path(os.environ.get(env_name, default)).expanduser()
    return (PROJECT_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()


TRIPS_DIR = _configured_path(
    "PATHSNAP_TRIPS_DIR", "data/Student folder/student_test"
)
PARTITION = os.environ.get("PATHSNAP_PARTITION", "student_test")
MODEL_FILE = _configured_path(
    "PATHSNAP_MODEL_FILE", "outputs/person2/models/final_model.joblib"
)
DECODER_CONFIG = _configured_path(
    "PATHSNAP_DECODER_CONFIG", "outputs/person2/models/decoder_config.json"
)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(title="PathSnap Route Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.state.predictor = None


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128)


def _get_predictor() -> RoutePredictor:
    predictor = app.state.predictor
    if predictor is None:
        if not MODEL_FILE.is_file():
            raise HTTPException(status_code=503, detail=f"Model file is not available: {MODEL_FILE}")
        if not DECODER_CONFIG.is_file():
            raise HTTPException(status_code=503, detail=f"Decoder config is not available: {DECODER_CONFIG}")
        try:
            predictor = RoutePredictor(MODEL_FILE, DECODER_CONFIG)
        except Exception as exc:
            LOG.exception("Could not load the selected model")
            raise HTTPException(status_code=503, detail="Could not load model or decoder configuration") from exc
        app.state.predictor = predictor
        LOG.info("Loaded model=%s artifact=%s decoder=%s", predictor.model_name, MODEL_FILE, DECODER_CONFIG)
    return predictor


def _trip_path(trip_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", trip_id):
        raise HTTPException(status_code=404, detail="Unknown trip id")
    candidate = (TRIPS_DIR / trip_id).resolve()
    if candidate.parent != TRIPS_DIR.resolve() or not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown trip {trip_id}")
    return candidate


@app.get("/health")
def health():
    predictor = _get_predictor()
    return {"status": "ok", "model": predictor.model_name}


@app.get("/trips")
def trips():
    if not TRIPS_DIR.is_dir():
        LOG.warning("Trip directory is not available: %s", TRIPS_DIR)
        return {"trips": []}
    trip_ids = sorted(
        path.name for path in TRIPS_DIR.iterdir()
        if path.is_dir()
        and (path / f"{path.name}.track").is_file()
        and (path / f"{path.name}.nodes").is_file()
        and (path / f"{path.name}.arcs").is_file()
    )
    LOG.info("GET /trips -> %d trip(s)", len(trip_ids))
    return {"trips": trip_ids}


@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    trip_dir = _trip_path(payload.trip_id)
    predictor = _get_predictor()
    LOG.info("POST /predict trip=%s from=%s", payload.trip_id, request.client.host if request.client else "unknown")
    try:
        # RoutePredictor loads only .track/.nodes/.arcs. Reference route is not
        # read until the separate aggregate evaluator below, after prediction.
        output, _ = predictor.predict(trip_dir, partition=PARTITION, include_network=True)
        output["evaluation"] = score_against_reference(trip_dir, output, PARTITION)
    except FileNotFoundError as exc:
        LOG.exception("Required trip or model data file was missing")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("Prediction failed for trip=%s", payload.trip_id)
        raise HTTPException(status_code=500, detail="Prediction failed; see API server log") from exc
    evaluation = output.get("evaluation")
    if evaluation is None:
        LOG.info("trip=%s points=%s; aggregate evaluation unavailable", payload.trip_id, output["n_points"])
    else:
        LOG.info(
            "trip=%s points=%s hmm_acc=%.4f classifier_acc=%.4f route_f1=%.4f breaks=%s",
            payload.trip_id, output["n_points"], evaluation["accuracy_hmm"],
            evaluation["accuracy_classifier_only"], evaluation["route_f1"],
            output["decoder"]["chain_breaks"],
        )
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    uvicorn.run("pathsnap_person2.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)


if __name__ == "__main__":
    main()
