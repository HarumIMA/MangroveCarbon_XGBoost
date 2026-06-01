"""
train.py
========
Latih model XGBoost untuk prediksi total_carbon_stock dan SIMPAN sebagai
artefak terserialisasi yang tetap (frozen).

Menyelesaikan MASALAH #1 (model tidak pernah disimpan):
  Menyimpan bundle artefak:
    - model.json    : booster XGBoost (format native, portabel antar-versi)
    - model.pkl     : bundle self-contained (joblib) = estimator + metadata,
                      untuk yang membutuhkan format pickle.
    - metadata.json : urutan fitur, info transform target, konstanta, median
                      untuk imputasi, versi pustaka, hash dataset, metrik.

Menyelesaikan MASALAH #2 (dua pipeline bertentangan):
  Memilih SATU pipeline final — log-transform target — karena distribusi
  total_carbon_stock sangat right-skewed (1.6 .. 5987). Model dilatih pada
  log1p(y) dan metadata mencatat "log1p" sehingga inference tahu harus
  meng-inverse dengan expm1(). Tidak ada lagi ambiguitas.

MODEL LEBIH ROBUST (untuk dataset kecil ~300 baris):
  - Regularisasi kuat (max_depth dangkal, min_child_weight, L1/L2, subsample &
    colsample) supaya pohon tidak menghafal sedikit sampel.
  - K-fold cross-validation pada porsi train untuk estimasi generalisasi yang
    stabil (mean ± std), bukan menebak dari satu hold-out 60-baris yang berisik.
  - Early stopping di tiap fold memilih jumlah pohon optimal secara otomatis;
    jumlah pohon final = rata-rata best-iteration antar-fold, lalu model final
    dilatih ulang memakai SELURUH porsi train (memaksimalkan data).
  Hasilnya: booster final berisi tepat n_estimators pohon, jadi predict.py
  tidak perlu berubah.

Cara pakai:
    python train.py --data /path/ke/civ_mangrove_biomass_dataset.csv \
                    --out  ./artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from features import FEATURE_ORDER, RAW_BANDS, SAVI_L, TARGET, build_feature_matrix

# Pipeline target yang DIPILIH untuk produksi.
TARGET_TRANSFORM = "log1p"  # latih pada log1p(y); inverse dengan expm1()

# --- Konfigurasi pembagian data (80% latih / 20% uji) ----------------------
# Dikunci sebagai konstanta bernama (bukan angka ajaib) agar pembagian 80/20
# konsisten & reprodusibel di seluruh proyek; nilai ini juga dicatat ke
# metadata dan dipakai ulang oleh smoke_test untuk evaluasi hold-out.
TEST_SIZE = 0.20          # 20% untuk test, sisanya (80%) untuk training
SPLIT_RANDOM_STATE = 42   # seed agar pembagian selalu sama

# Hyperparameter (ruang log-transform), disetel untuk ROBUST di data kecil.
# n_estimators di sini hanyalah BATAS ATAS; jumlah pohon final ditentukan oleh
# early stopping pada cross-validation (lihat _cross_validate).
MODEL_PARAMS = {
    "n_estimators": 3000,       # batas atas; early stopping yang menentukan
    "learning_rate": 0.02,      # lebih kecil + lebih banyak pohon -> lebih halus
    "max_depth": 4,             # pohon dangkal -> kurang overfit di data kecil
    "min_child_weight": 5.0,    # daun butuh cukup sampel -> cegah hafalan
    "subsample": 0.8,           # bagging baris tiap pohon
    "colsample_bytree": 0.8,    # bagging fitur tiap pohon
    "reg_alpha": 0.5,           # regularisasi L1
    "reg_lambda": 2.0,          # regularisasi L2
    "gamma": 0.1,               # ambang minimal penurunan loss untuk split
    "random_state": 42,
}

# Konfigurasi cross-validation & early stopping (estimasi robust + pilih jumlah pohon).
N_CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
MIN_ESTIMATORS = 50            # batas bawah pengaman bila early stopping terlalu dini


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _forward_transform(y: np.ndarray) -> np.ndarray:
    return np.log1p(y) if TARGET_TRANSFORM == "log1p" else y


def _inverse_transform(y: np.ndarray) -> np.ndarray:
    return np.expm1(y) if TARGET_TRANSFORM == "log1p" else y


def _cross_validate(Xr_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    """
    K-fold CV pada porsi TRAIN untuk:
      1. Estimasi generalisasi yang stabil (mean ± std MAE/RMSE/R2).
      2. Memilih jumlah pohon optimal via early stopping di tiap fold.

    Anti-kebocoran: median imputasi dihitung ULANG dari fold-train saja pada
    setiap lipatan, persis seperti pada pipeline final.
    """
    Xr_train = Xr_train.reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=float)

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SPLIT_RANDOM_STATE)
    maes, rmses, r2s, best_iters = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xr_train), start=1):
        Xr_tr, Xr_va = Xr_train.iloc[tr_idx], Xr_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        medians = build_feature_matrix(Xr_tr, medians=None).median(numeric_only=True).to_dict()
        X_tr = build_feature_matrix(Xr_tr, medians=medians)
        X_va = build_feature_matrix(Xr_va, medians=medians)

        m = XGBRegressor(**MODEL_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        m.fit(
            X_tr, _forward_transform(y_tr),
            eval_set=[(X_va, _forward_transform(y_va))],
            verbose=False,
        )

        pred_va = _inverse_transform(m.predict(X_va))
        maes.append(float(mean_absolute_error(y_va, pred_va)))
        rmses.append(float(np.sqrt(mean_squared_error(y_va, pred_va))))
        r2s.append(float(r2_score(y_va, pred_va)))
        # best_iteration adalah indeks; jumlah pohon = indeks + 1.
        best_iters.append(int(m.best_iteration) + 1)
        print(f"    fold {fold}/{N_CV_FOLDS}: MAE={maes[-1]:.2f}  "
              f"R2={r2s[-1]:.4f}  n_trees={best_iters[-1]}")

    chosen = max(MIN_ESTIMATORS, int(round(float(np.mean(best_iters)))))
    return {
        "n_folds": N_CV_FOLDS,
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "best_iterations": best_iters,
        "chosen_n_estimators": chosen,
    }


def train(data_path: str, out_dir: str) -> dict:
    data_path = Path(data_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Muat data ---------------------------------------------------------
    df = pd.read_csv(data_path)
    needed = RAW_BANDS + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset kekurangan kolom: {missing}")

    # Buang baris tanpa target (saat training boleh dibuang; saat serving tidak).
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # --- Bangun fitur (tanpa medians dulu, agar median dihitung dari training)
    X_all_raw = df[RAW_BANDS]
    y_all = df[TARGET].to_numpy(dtype=float)

    # Bagi 80% latih / 20% uji pada data MENTAH supaya median dihitung hanya
    # dari fold train (mencegah kebocoran statistik imputasi ke test).
    Xr_train, Xr_test, y_train, y_test = train_test_split(
        X_all_raw, y_all, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )

    # Simpan pembagian 80/20 sebagai artefak transparan & reprodusibel:
    # train.csv (80%) dan test.csv (20%), masing-masing band mentah + target.
    # Ini membuat hold-out eksplisit dan bisa dipakai ulang untuk evaluasi.
    train_split = Xr_train.copy()
    train_split[TARGET] = y_train
    test_split = Xr_test.copy()
    test_split[TARGET] = y_test
    train_split.to_csv(out / "train.csv", index=False)
    test_split.to_csv(out / "test.csv", index=False)

    # Median fitur dihitung dari TRAIN saja, untuk imputasi saat inference.
    feat_train_for_median = build_feature_matrix(Xr_train, medians=None)
    medians = feat_train_for_median.median(numeric_only=True).to_dict()

    # Bangun matriks fitur final memakai median training.
    X_train = build_feature_matrix(Xr_train, medians=medians)
    X_test = build_feature_matrix(Xr_test, medians=medians)

    # --- Cross-validation pada porsi TRAIN (estimasi robust + pilih #pohon) -
    print(f"  Cross-validation {N_CV_FOLDS}-fold pada porsi train ...")
    cv = _cross_validate(Xr_train, y_train)
    chosen_estimators = cv["chosen_n_estimators"]
    print(f"    CV: R2={cv['r2_mean']:.4f}±{cv['r2_std']:.4f}  "
          f"MAE={cv['mae_mean']:.2f}±{cv['mae_std']:.2f}  "
          f"-> n_estimators={chosen_estimators}")

    # --- Latih model FINAL pada SELURUH porsi train (ruang log1p) ----------
    # Memakai jumlah pohon hasil CV (tanpa early stopping), sehingga booster
    # berisi tepat `chosen_estimators` pohon -> inference (predict.py) konsisten.
    y_train_t = _forward_transform(y_train)
    final_params = {**MODEL_PARAMS, "n_estimators": chosen_estimators}
    model = XGBRegressor(**final_params)
    model.fit(X_train, y_train_t)

    # --- Evaluasi di ruang ASLI (inverse dulu) -----------------------------
    pred_test = _inverse_transform(model.predict(X_test))
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "r2": float(r2_score(y_test, pred_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    # --- SIMPAN ARTEFAK (Masalah #1) ---------------------------------------
    model_path = out / "model.json"
    model.get_booster().save_model(str(model_path))  # format native, portabel

    metadata = {
        "model_type": "XGBRegressor",
        "model_files": {                         # dua format model tersedia
            "native": "model.json",              # portabel antar-versi (disarankan)
            "pickle": "model.pkl",               # bundle self-contained (joblib)
        },
        "target": TARGET,
        "target_transform": TARGET_TRANSFORM,   # <- kunci konsistensi (Masalah #2)
        "feature_order": FEATURE_ORDER,
        "raw_bands": RAW_BANDS,
        "savi_L": SAVI_L,
        "feature_medians": medians,             # untuk imputasi saat inference
        "model_params": final_params,           # params final (n_estimators dari CV)
        "cv": cv,                               # estimasi robust K-fold + #pohon terpilih
        "split": {                              # pembagian 80% latih / 20% uji
            "method": "random_holdout",
            "test_size": TEST_SIZE,
            "random_state": SPLIT_RANDOM_STATE,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_file": "train.csv",
            "test_file": "test.csv",
        },
        "metrics": metrics,
        "library_versions": {
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "dataset": {
            "filename": data_path.name,
            "sha256": _sha256(data_path),
            "n_rows": int(len(df)),
        },
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Simpan juga sebagai .pkl (bundle self-contained, joblib) ----------
    # Berisi estimator terlatih + seluruh info yang dibutuhkan inference,
    # sehingga model.pkl saja sudah cukup tanpa metadata.json terpisah.
    # Catatan: pickle terikat versi pustaka; model.json tetap format utama
    # yang portabel antar-versi.
    pkl_path = out / "model.pkl"
    bundle = {
        "model": model,                       # XGBRegressor terlatih (memuat booster)
        "feature_order": FEATURE_ORDER,
        "raw_bands": RAW_BANDS,
        "target": TARGET,
        "target_transform": TARGET_TRANSFORM,
        "feature_medians": medians,
        "savi_L": SAVI_L,
        "metadata": metadata,                 # salinan metadata lengkap
    }
    joblib.dump(bundle, pkl_path)

    print("=" * 60)
    print("TRAINING SELESAI — artefak tersimpan")
    print("=" * 60)
    print(f"  Model     : {model_path}")
    print(f"  Model PKL : {pkl_path}")
    print(f"  Metadata  : {out / 'metadata.json'}")
    print(f"  Split     : {len(y_train)} latih / {len(y_test)} uji "
          f"({int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)})")
    print(f"  Transform : {TARGET_TRANSFORM}")
    print(f"  Pohon     : {chosen_estimators} (dipilih via CV early stopping)")
    print(f"  CV  R2    : {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f} ({N_CV_FOLDS}-fold)")
    print(f"  CV  MAE   : {cv['mae_mean']:.2f} ± {cv['mae_std']:.2f}")
    print("  --- hold-out 20% ---")
    print(f"  MAE  : {metrics['mae']:.2f}")
    print(f"  RMSE : {metrics['rmse']:.2f}")
    print(f"  R2   : {metrics['r2']:.4f}")
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Latih & simpan model carbon stock.")
    ap.add_argument("--data", required=True, help="Path CSV dataset.")
    ap.add_argument("--out", default="./artifacts", help="Folder output artefak.")
    args = ap.parse_args()
    train(args.data, args.out)
