"""
features.py
===========
Sumber kebenaran TUNGGAL untuk feature engineering.

Modul ini diimpor baik oleh training (train.py) maupun serving (predict.py),
sehingga tidak mungkin terjadi training-serving skew: rumus yang dipakai saat
melatih dan saat memprediksi dijamin identik.

Menyelesaikan MASALAH #3 (pembagian nol):
  NDVI/NDWI/SAVI membagi penyebut yang bisa bernilai 0 pada piksel air, awan,
  atau bayangan. Di sini setiap rasio dihitung dengan penjaga (epsilon) dan
  hasil inf/NaN diganti nilai netral, sehingga model tidak pernah menerima
  inf/NaN saat inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Konstanta yang HARUS sama antara training & serving --------------------

SAVI_L = 0.5  # koefisien koreksi tanah pada SAVI

# Band mentah yang wajib ada pada input sebelum feature engineering.
RAW_BANDS = [
    "longitude", "latitude",
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "VV", "VH",
]

# Indeks turunan yang dihitung modul ini.
DERIVED_FEATURES = ["NDVI", "NDWI", "SAVI"]

# Urutan fitur final yang masuk ke model — urutan ini DIKUNCI.
FEATURE_ORDER = [
    "longitude", "latitude",
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "VV", "VH",
    "NDVI", "NDWI", "SAVI",
]

TARGET = "total_carbon_stock"

# Epsilon kecil untuk mencegah pembagian nol secara numerik.
_EPS = 1e-8


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Bagi numerator/denominator dengan aman.

    - Saat |denominator| < _EPS, hasilnya didefinisikan 0 (konvensi umum untuk
      indeks vegetasi pada piksel tanpa sinyal, mis. air murni).
    - Sisa inf/-inf/NaN yang mungkin lolos diganti 0.
    """
    denom = denominator.where(denominator.abs() >= _EPS, np.nan)
    ratio = numerator / denom
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan NDVI, NDWI, SAVI ke salinan df. Tidak mengubah df asli.

    Rumus identik dengan notebook, tetapi memakai _safe_ratio sehingga aman
    terhadap penyebut nol.
    """
    out = df.copy()

    # NDVI = (B8 - B4) / (B8 + B4)   -> rentang teori [-1, 1]
    out["NDVI"] = _safe_ratio(out["B8"] - out["B4"], out["B8"] + out["B4"]).clip(-1.0, 1.0)

    # NDWI = (B3 - B8) / (B3 + B8)   -> rentang teori [-1, 1]
    out["NDWI"] = _safe_ratio(out["B3"] - out["B8"], out["B3"] + out["B8"]).clip(-1.0, 1.0)

    # SAVI = ((B8 - B4) * (1 + L)) / (B8 + B4 + L)
    out["SAVI"] = _safe_ratio((out["B8"] - out["B4"]) * (1.0 + SAVI_L),
                              out["B8"] + out["B4"] + SAVI_L)

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    medians: dict | None = None,
) -> pd.DataFrame:
    """
    Ubah DataFrame mentah menjadi matriks fitur siap-model.

    Langkah:
      1. Validasi band mentah tersedia.
      2. Hitung indeks spektral secara aman (NDVI/NDWI/SAVI).
      3. Imputasi nilai hilang/inf pada band mentah dengan `medians` bila
         disediakan (median dari data training, disimpan dalam metadata model).
         Tanpa medians, NaN pada band mentah diisi 0 sebagai jaring pengaman.
      4. Susun ulang kolom sesuai FEATURE_ORDER yang dikunci.

    Parameter
    ---------
    df : pd.DataFrame
        Harus memuat seluruh kolom dalam RAW_BANDS.
    medians : dict | None
        {nama_fitur: nilai_median} dari training, dipakai untuk imputasi saat
        inference. Saat training, biarkan None (median dihitung di train.py).

    Mengembalikan
    -------------
    pd.DataFrame dengan kolom persis FEATURE_ORDER, tanpa NaN/inf.
    """
    missing = [c for c in RAW_BANDS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom band mentah hilang dari input: {missing}")

    # Bersihkan inf pada band mentah lebih dulu agar rasio tidak meledak.
    feat = df[RAW_BANDS].replace([np.inf, -np.inf], np.nan)

    feat = add_spectral_indices(feat)

    # Imputasi NaN. Saat serving pakai median training; saat training, medians
    # belum tersedia sehingga NaN sisa diisi 0 (akan langka karena _safe_ratio
    # sudah menangani indeks, dan band mentah seharusnya lengkap di training).
    if medians is not None:
        feat = feat.fillna(value=medians)
    feat = feat.fillna(0.0)

    # Kunci urutan kolom — XGBoost memetakan fitur berdasarkan posisi.
    feat = feat[FEATURE_ORDER]

    # Jaring pengaman terakhir: pastikan benar-benar bersih.
    if not np.isfinite(feat.to_numpy()).all():
        raise ValueError("Matriks fitur masih memuat inf/NaN setelah pembersihan.")

    return feat
