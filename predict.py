"""
predict.py
==========
Muat artefak terlatih dan lakukan prediksi yang aman & konsisten.

Mengikat ketiga perbaikan menjadi satu jalur inference:
  #1  Memuat model terserialisasi (model.json) — tidak melatih ulang.
  #2  Membaca target_transform dari metadata dan meng-inverse dengan benar
      (expm1 bila log1p), sehingga prediksi berada di skala asli.
  #3  Membangun fitur lewat features.build_feature_matrix dengan median
      training untuk imputasi, sehingga input penyebut-nol / NaN tidak
      pernah membuat model error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import build_feature_matrix


class CarbonStockModel:
    """Pembungkus inference yang memuat bundle artefak satu kali."""

    def __init__(self, artifact_dir: str):
        art = Path(artifact_dir)
        with open(art / "metadata.json") as f:
            self.meta = json.load(f)

        self.booster = xgb.Booster()
        self.booster.load_model(str(art / "model.json"))

        self.feature_order = self.meta["feature_order"]
        self.target_transform = self.meta["target_transform"]
        self.medians = self.meta["feature_medians"]

    @classmethod
    def from_pickle(cls, pkl_path: str) -> "CarbonStockModel":
        """
        Muat model dari bundle .pkl (joblib) yang dihasilkan train.py.

        Alternatif dari memuat model.json + metadata.json; bundle .pkl sudah
        self-contained. Jalur inference (predict) tetap sama persis.
        """
        import joblib

        obj = cls.__new__(cls)  # lewati __init__ (tak perlu baca file json)
        bundle = joblib.load(pkl_path)
        obj.meta = bundle.get("metadata", {})
        obj.feature_order = bundle["feature_order"]
        obj.target_transform = bundle["target_transform"]
        obj.medians = bundle["feature_medians"]
        # Pakai booster dari estimator terlatih -> predict() tak perlu berubah.
        obj.booster = bundle["model"].get_booster()
        return obj

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            return np.expm1(y)
        return y

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        """
        Prediksi total_carbon_stock dari DataFrame band mentah.

        `raw` harus memuat kolom RAW_BANDS (longitude, latitude, B2..B12,
        VV, VH). Baris dengan band hilang/inf akan diimputasi otomatis,
        bukan dibuang — sesuai kebutuhan produksi.
        """
        X = build_feature_matrix(raw, medians=self.medians)
        dmatrix = xgb.DMatrix(X.to_numpy(), feature_names=self.feature_order)
        pred_t = self.booster.predict(dmatrix)
        pred = self._inverse(pred_t)
        # Stok karbon tidak mungkin negatif; jepit di 0.
        return np.clip(pred, 0.0, None)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prediksi carbon stock dari CSV.")
    ap.add_argument("--artifacts", required=True, help="Folder artefak.")
    ap.add_argument("--input", required=True, help="CSV band mentah.")
    ap.add_argument("--output", default=None, help="CSV hasil (opsional).")
    args = ap.parse_args()

    model = CarbonStockModel(args.artifacts)
    df_in = pd.read_csv(args.input)
    df_in["predicted_carbon"] = model.predict(df_in)

    if args.output:
        df_in.to_csv(args.output, index=False)
        print(f"Hasil ditulis ke {args.output}")
    else:
        print(df_in[["predicted_carbon"]].describe())
