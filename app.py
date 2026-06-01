"""
app.py
======
REST API (FastAPI) yang membungkus CarbonStockModel untuk serving.

Memuat bundle artefak (model.json + metadata.json) SATU KALI saat startup,
lalu melayani prediksi via HTTP. Tidak ada training di sini — murni inference,
konsisten dengan jalur di predict.py (transform log1p -> expm1, imputasi median,
penjepitan >= 0, aman terhadap penyebut nol / NaN).

Endpoint:
  GET  /          : info ringkas API & status model
  GET  /health    : health check (200 bila model siap, 503 bila belum)
  GET  /metadata  : metadata model (metrik, split, versi pustaka, dll.)
  POST /predict   : prediksi batch dari daftar record band mentah

Jalankan lokal:
    uvicorn app:app --host 0.0.0.0 --port 8000
Lalu buka dokumentasi interaktif di  http://localhost:8000/docs

Folder artefak diambil dari env ARTIFACT_DIR (default: ./artifacts), sehingga
di container bisa di-mount/override tanpa mengubah kode.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from features import RAW_BANDS
from predict import CarbonStockModel

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")

# State proses: model dimuat sekali, dipakai ulang lintas-request.
_state: dict = {"model": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Muat artefak sekali saat startup; lepaskan saat shutdown."""
    try:
        _state["model"] = CarbonStockModel(ARTIFACT_DIR)
        _state["load_error"] = None
    except Exception as exc:  # noqa: BLE001 — laporkan via /health, jangan crash startup
        _state["model"] = None
        _state["load_error"] = f"{type(exc).__name__}: {exc}"
    yield
    _state["model"] = None


app = FastAPI(
    title="Carbon Stock Model API",
    description="Inference total_carbon_stock dari band Sentinel (optik + SAR).",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Skema I/O --------------------------------------------------------------
class BandInput(BaseModel):
    """
    Satu observasi band mentah. Semua field opsional: nilai yang hilang (null)
    akan diimputasi otomatis dengan median training (sesuai predict.py),
    sehingga input parsial tetap aman di produksi.
    """
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    B2: Optional[float] = None
    B3: Optional[float] = None
    B4: Optional[float] = None
    B5: Optional[float] = None
    B6: Optional[float] = None
    B7: Optional[float] = None
    B8: Optional[float] = None
    B8A: Optional[float] = None
    B11: Optional[float] = None
    B12: Optional[float] = None
    VV: Optional[float] = None
    VH: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "longitude": -5.5801, "latitude": 5.1047,
                "B2": 0.0537, "B3": 0.0730, "B4": 0.0607, "B5": 0.1049,
                "B6": 0.2407, "B7": 0.2916, "B8": 0.2831, "B8A": 0.3104,
                "B11": 0.1370, "B12": 0.0733, "VV": -7.28, "VH": -14.54,
            }
        }
    }


class PredictRequest(BaseModel):
    records: list[BandInput] = Field(..., min_length=1,
                                     description="Daftar observasi band mentah (>=1).")


class PredictResponse(BaseModel):
    predictions: list[float] = Field(..., description="Prediksi total_carbon_stock (skala asli).")
    n: int = Field(..., description="Jumlah baris yang diprediksi.")
    target: str = "total_carbon_stock"


# --- Endpoint ---------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Carbon Stock Model API",
        "version": "1.0.0",
        "model_ready": _state["model"] is not None,
        "docs": "/docs",
        "endpoints": ["/health", "/metadata", "/predict"],
    }


@app.get("/health")
def health():
    if _state["model"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_loaded": False,
                     "artifact_dir": ARTIFACT_DIR, "error": _state["load_error"]},
        )
    return {"status": "ok", "model_loaded": True, "artifact_dir": ARTIFACT_DIR}


@app.get("/metadata")
def metadata():
    model = _state["model"]
    if model is None:
        return JSONResponse(status_code=503,
                            content={"error": "Model belum dimuat", "detail": _state["load_error"]})
    # Jangan kirim feature_medians yang panjang; ringkas saja yang relevan.
    meta = model.meta
    return {
        "model_type": meta.get("model_type"),
        "target": meta.get("target"),
        "target_transform": meta.get("target_transform"),
        "feature_order": meta.get("feature_order"),
        "metrics": meta.get("metrics"),
        "cv": meta.get("cv"),
        "split": meta.get("split"),
        "library_versions": meta.get("library_versions"),
        "trained_at_utc": meta.get("trained_at_utc"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = _state["model"]
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model belum dimuat", "detail": _state["load_error"]},
        )
    # Susun DataFrame dengan SEMUA kolom band (urutan dikunci); null -> NaN -> diimputasi.
    df = pd.DataFrame([r.model_dump() for r in req.records])[RAW_BANDS]
    preds = model.predict(df)
    return PredictResponse(predictions=[float(p) for p in preds], n=int(len(preds)))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
