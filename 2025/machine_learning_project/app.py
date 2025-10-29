#!/usr/bin/env python3
from pathlib import Path
import pickle
from typing import Any, List
from fastapi import FastAPI, HTTPException
from models.Record import Record
from scripts.score_record import load_pipeline, prepare_input, score

app = FastAPI(title="ML pipeline service")

PIPELINE_PATH = Path("pipeline_v1.bin")

# def score(pipeline: Any, X) -> List[float]:
#     def _call(pX):
#         if hasattr(pipeline, "predict_proba"):
#             out = pipeline.predict_proba(pX)
#             try:
#                 if hasattr(out, "ndim") and out.ndim == 2 and out.shape[1] == 2:
#                     return out[:, 1].tolist()
#             except Exception:
#                 pass
#             try:
#                 return out.tolist()
#             except Exception:
#                 return list(out)
#         if hasattr(pipeline, "predict"):
#             out = pipeline.predict(pX)
#             try:
#                 return out.tolist()
#             except Exception:
#                 return list(out)
#         raise RuntimeError("Loaded object has neither predict_proba nor predict")

#     try:
#         return _call(X)
#     except Exception:
#         try:
#             import pandas as pd  # type: ignore
#             if isinstance(X, pd.DataFrame):
#                 return _call(X.to_dict(orient="records"))
#             if isinstance(X, list):
#                 return _call(pd.DataFrame(X))
#         except Exception:
#             pass
#         raise


# Load pipeline at startup
try:
    PIPELINE = load_pipeline(PIPELINE_PATH)
except Exception as e:
    PIPELINE = None
    load_error = str(e)
else:
    load_error = None


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": PIPELINE is not None, "load_error": load_error}


@app.post("/predict")
def predict(record: Record):
    if PIPELINE is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")
    X = prepare_input(record.dict())
    try:
        scores = score(PIPELINE, X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")
    # return single score for single record
    return {"record": record.dict(), "score": scores[0] if isinstance(scores, (list, tuple)) else scores}