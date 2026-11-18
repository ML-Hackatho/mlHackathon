"""TEST API for developing the UI - follows INTERFACE.md.

Person 2 owns the real API. This file exists so Person 3 can test the UI end-to-end,
and so Person 2 has a working reference for the agreed request/response format.

Run (from the project root):
    pip install -r requirements.txt
    PATHSNAP_EVAL_DIR=data/student_test uvicorn dev_api.test_api:app --host 0.0.0.0 --port 8000

Every request is logged to the console (the evaluator must see these logs on Laptop 1).
"""
import logging
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.inference import RoutePredictor, score_against_reference  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap-api")

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "eval"))
MODEL = os.environ.get("PATHSNAP_MODEL_FILE", os.path.join(ROOT, "models", "mlp.joblib"))
DECODER = os.environ.get("PATHSNAP_DECODER_CONFIG", os.path.join(ROOT, "models", "decoder_config.json"))

app = FastAPI(title="PathSnap test API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = RoutePredictor(MODEL, DECODER)
log.info("Loaded model %s, decoder %s, trips from %s", MODEL, DECODER, EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.model_name}


@app.get("/trips")
def trips():
    t = list_trips(EVAL_DIR)
    log.info("GET /trips -> %d trips", len(t))
    return {"trips": t}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    d = os.path.join(EVAL_DIR, req.trip_id)
    log.info("POST /predict trip=%s - running pipeline", req.trip_id)
    t0 = time.time()
    out, trip = predictor.predict(d)                          # prediction never reads .route
    out["evaluation"], _ = score_against_reference(d, out)   # server-side scoring only (numbers)
    e = out["evaluation"]
    log.info("trip=%s points=%d acc_hmm=%.4f acc_clf=%.4f route_f1=%.4f breaks=%d time=%.2fs",
             req.trip_id, out["n_points"], e["accuracy_hmm"], e["accuracy_classifier_only"],
             e["route_f1"], out["decoder"]["chain_breaks"], time.time() - t0)
    return out
