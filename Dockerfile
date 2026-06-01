# Image serving untuk Carbon Stock Model API (FastAPI + XGBoost).
# Python 3.13 menyamai versi yang dipakai saat melatih (lihat metadata.json),
# agar format model.json kompatibel.
FROM python:3.13-slim

# Hindari .pyc & buffering log agar log container muncul real-time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/app/artifacts \
    PORT=8000

WORKDIR /app

# 1) Dependensi dulu (layer cache: tidak re-install bila hanya kode berubah).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Kode sumber yang dibutuhkan untuk inference.
COPY features.py predict.py app.py ./

# 3) Artefak model terlatih (HARUS sudah ada: jalankan train.py lebih dulu).
#    Bisa juga di-override saat runtime dengan volume + env ARTIFACT_DIR.
COPY artifacts ./artifacts

# Jalankan sebagai user non-root (praktik keamanan dasar).
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check sederhana memakai endpoint /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')" || exit 1

# Satu worker uvicorn; skalakan via replika container, bukan banyak worker
# (model XGBoost in-memory per proses).
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
