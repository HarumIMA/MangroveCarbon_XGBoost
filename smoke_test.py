"""
smoke_test.py
=============
Uji end-to-end memakai DATA ASLI dari Kaggle, bukan data sintetis:

  "Mangrove Biomass and Carbon Stocks Dataset" (salomonkouassi)
  https://www.kaggle.com/datasets/salomonkouassi/mangrove-biomass-and-carbon-stocks-dataset

Dataset (civ_mangrove_biomass_dataset.csv) memuat seluruh band yang dibutuhkan
model: longitude, latitude, B2..B12, VV, VH, dan target total_carbon_stock.
File akan diunduh otomatis ke ./data bila belum tersedia (endpoint publik
Kaggle, tanpa kredensial). Bila pengunduhan otomatis gagal (mis. tanpa
internet), unduh manual dari tautan di atas dan letakkan CSV di ./data.

Memverifikasi ketiga perbaikan, sekarang di atas data nyata:
  - Masalah #1: artefak tersimpan & bisa dimuat ulang.
  - Masalah #2: train log1p -> inference expm1, skala prediksi benar.
  - Masalah #3: input penyebut-nol (B8=B4=0, dst.) tidak membuat model error.

Karena data nyata, metrik (MAE/RMSE/R2) yang dicetak train.py kini BERMAKNA.
"""

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from features import RAW_BANDS, TARGET, build_feature_matrix
from predict import CarbonStockModel
from train import TEST_SIZE, train

# --- Sumber data asli (Kaggle) ---------------------------------------------
KAGGLE_SLUG = "salomonkouassi/mangrove-biomass-and-carbon-stocks-dataset"
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_SLUG}"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "civ_mangrove_biomass_dataset.csv"

TMP = ROOT / "_smoke"          # folder kerja sementara (lintas-OS)
ART = TMP / "artifacts"


def ensure_dataset() -> Path:
    """Pastikan CSV dataset asli tersedia; unduh dari Kaggle bila belum ada."""
    if CSV_PATH.exists():
        print(f"Dataset ditemukan di cache lokal: {CSV_PATH}")
        return CSV_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh dataset dari Kaggle: {KAGGLE_SLUG} ...")
    req = urllib.request.Request(KAGGLE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:          # noqa: S310 (URL tetap/dipercaya)
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(DATA_DIR)

    if CSV_PATH.exists():
        return CSV_PATH
    # Fallback: ambil CSV pertama yang terekstrak bila nama berbeda.
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("Tidak menemukan CSV dalam arsip Kaggle.")
    return csvs[0]


def main():
    # --- Muat data asli ----------------------------------------------------
    csv = ensure_dataset()
    df = pd.read_csv(csv)
    print(f"Dataset asli dimuat: {len(df)} baris, {df.shape[1]} kolom")

    needed = RAW_BANDS + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    assert not missing, f"Dataset kekurangan kolom yang dibutuhkan model: {missing}"
    print("OK  semua kolom band + target tersedia di dataset asli")

    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)

    print("\n--- Latih & simpan artefak pada data asli (Masalah #1 & #2) ---")
    meta = train(str(csv), str(ART))

    assert (ART / "model.json").exists(), "model.json tidak tersimpan!"
    assert (ART / "metadata.json").exists(), "metadata.json tidak tersimpan!"
    assert meta["target_transform"] == "log1p", "pipeline target tidak terkunci!"
    print("OK  artefak tersimpan & transform terkunci = log1p")

    print("\n--- Verifikasi pembagian 80% latih / 20% uji ---")
    split = meta["split"]
    assert (ART / split["train_file"]).exists(), "train.csv tidak tersimpan!"
    assert (ART / split["test_file"]).exists(), "test.csv tidak tersimpan!"
    n_total = split["n_train"] + split["n_test"]
    test_frac = split["n_test"] / n_total
    assert abs(test_frac - TEST_SIZE) < 0.02, (
        f"proporsi test {test_frac:.2%} menyimpang dari {TEST_SIZE:.0%}!"
    )
    # train.csv + test.csv harus menutup seluruh baris berlabel, tanpa tumpang tindih.
    train_df = pd.read_csv(ART / split["train_file"])
    test_df = pd.read_csv(ART / split["test_file"])
    n_labeled = int(df[TARGET].notna().sum())
    assert len(train_df) + len(test_df) == n_labeled, "split tidak menutup semua baris!"
    print(f"OK  pembagian 80/20 valid: {split['n_train']} latih / {split['n_test']} uji "
          f"({1 - test_frac:.0%}/{test_frac:.0%})")

    print("\n--- Muat ulang model dari disk (Masalah #1) ---")
    model = CarbonStockModel(str(ART))
    print("OK  model dimuat tanpa training ulang")

    print("\n--- Evaluasi hold-out pada 20% test (Masalah #2) ---")
    # Prediksi HANYA pada test set (20%) yang tidak ikut dilatih — evaluasi jujur.
    preds = model.predict(test_df[RAW_BANDS])
    y_true = test_df[TARGET].to_numpy(dtype=float)
    print(f"    rentang prediksi : [{preds.min():.1f}, {preds.max():.1f}]")
    print(f"    rentang target   : [{y_true.min():.1f}, {y_true.max():.1f}]")
    assert preds.min() >= 0, "prediksi negatif — transform/inverse salah!"
    assert preds.max() > 50, "prediksi terlalu kecil — inverse expm1 tak jalan?"
    print("OK  prediksi berada di skala asli (bukan skala log)")

    # Metrik yang dihitung ulang di sini harus cocok dengan yang dicatat train.py
    # (membuktikan metadata.metrics memang dari hold-out 20%, bukan dari data latih).
    mae_holdout = float(mean_absolute_error(y_true, preds))
    r2_holdout = float(r2_score(y_true, preds))
    m = meta["metrics"]
    assert abs(mae_holdout - m["mae"]) < 1.0, (
        f"MAE hold-out ({mae_holdout:.2f}) tak cocok dengan metadata ({m['mae']:.2f})!"
    )
    print(f"OK  metrik hold-out cocok dengan metadata: "
          f"MAE={mae_holdout:.2f}  R2={r2_holdout:.4f}")

    print("\n--- Robustness: cross-validation & jumlah pohon terpilih ---")
    cv = meta["cv"]
    assert cv["n_folds"] >= 2, "CV harus >= 2 fold!"
    assert len(cv["best_iterations"]) == cv["n_folds"], "best_iterations tak lengkap!"
    # Jumlah pohon final harus hasil pemilihan CV (bukan batas atas MODEL_PARAMS).
    assert meta["model_params"]["n_estimators"] == cv["chosen_n_estimators"], (
        "n_estimators final tidak sama dengan pilihan CV!"
    )
    print(f"    CV {cv['n_folds']}-fold : R2={cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}  "
          f"MAE={cv['mae_mean']:.2f} ± {cv['mae_std']:.2f}")
    print(f"    pohon final : {cv['chosen_n_estimators']} "
          f"(best per-fold: {cv['best_iterations']})")
    print("OK  jumlah pohon dipilih otomatis & estimasi generalisasi tersedia")

    print("\n--- Input JAHAT: penyebut nol & NaN (Masalah #3) ---")
    evil = pd.DataFrame([
        {**{c: 0.0 for c in RAW_BANDS}},                         # semua nol -> NDVI 0/0
        {**{c: 0.0 for c in RAW_BANDS}, "B8": 0.3, "B4": -0.3},  # B8+B4 = 0
        {**{c: 0.1 for c in RAW_BANDS}, "B3": np.nan},           # NaN mentah
        {**{c: 0.1 for c in RAW_BANDS}, "B8": np.inf},           # inf mentah
    ])
    # Pastikan feature builder benar-benar bersih sebelum model.
    X_evil = build_feature_matrix(evil, medians=model.medians)
    assert np.isfinite(X_evil.to_numpy()).all(), "masih ada inf/NaN di fitur!"
    evil_preds = model.predict(evil)
    assert np.isfinite(evil_preds).all(), "prediksi inf/NaN pada input jahat!"
    print(f"    prediksi input jahat : {np.round(evil_preds, 1).tolist()}")
    print("OK  tidak ada inf/NaN; model tidak error pada penyebut nol")

    m = meta["metrics"]
    print("\n========================================================")
    print("SEMUA PERIKSA LULUS — model robust pada data ASLI.")
    print(f"  CV {cv['n_folds']}-fold   : R2={cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}  "
          f"MAE={cv['mae_mean']:.2f} ± {cv['mae_std']:.2f}")
    print(f"  Hold-out 20% : MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.4f}")
    print(f"  (n_train={m['n_train']}, n_test={m['n_test']}, "
          f"pohon={cv['chosen_n_estimators']})")
    print("========================================================")


if __name__ == "__main__":
    main()
