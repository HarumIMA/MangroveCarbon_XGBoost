# Carbon Stock Model — perbaikan kesiapan deploy


| # | Masalah | Solusi | Lokasi |
|---|---------|--------|--------|
| 1 | Model tidak pernah disimpan | Serialisasi ke bundle `model.json` + `metadata.json` (format native, portabel) | `train.py` |
| 2 | Dua pipeline bertentangan (log vs mentah) | Dikunci ke satu pipeline: `log1p` saat latih, `expm1` saat inference; tercatat di metadata | `train.py`, `predict.py` |
| 3 | Pembagian nol di NDVI/NDWI/SAVI | `_safe_ratio` + clip + imputasi median; tidak ada lagi inf/NaN | `features.py` |

## Struktur
- `features.py` — feature engineering aman (sumber kebenaran tunggal training & serving)
- `train.py`    — latih + simpan artefak
- `predict.py`  — muat artefak + inference aman
- `app.py`      — REST API (FastAPI) yang membungkus `predict.py`
- `Dockerfile`  — image serving siap-deploy
- `smoke_test.py` — uji end-to-end memakai **data asli Kaggle** (diunduh otomatis ke `./data`)

## Pembagian data (80% latih / 20% uji)
`train.py` membagi dataset **80% untuk training, 20% untuk test** memakai
`train_test_split` dengan seed tetap (`TEST_SIZE=0.20`, `SPLIT_RANDOM_STATE=42`),
sehingga selalu reprodusibel. Pembagian disimpan eksplisit sebagai artefak:
- `train.csv` — 80% (band mentah + target)
- `test.csv`  — 20% hold-out (band mentah + target)

Median imputasi dan metrik (MAE/RMSE/R²) dihitung **hanya dari fold yang benar**
(median dari train, metrik dari test) untuk mencegah kebocoran. Rincian
pembagian dicatat di `metadata.json` pada field `split`.

## Data cleaning & EDA (ringkas)
Pemeriksaan dataset (300 baris) menunjukkan **data sudah bersih**, sehingga
cleaning tambahan tidak diperlukan:
- **Missing value:** 0 pada seluruh band & target.
- **Duplikat:** 0 baris.
- **Target right-skewed** (skew ≈ 2.9; rentang 1.6–5987) → ditangani lewat
  transform `log1p` saat latih dan `expm1` saat inference (lihat `train.py`).
- **Indeks aman** (NDVI/NDWI/SAVI) dihitung dengan penjaga pembagian-nol di
  `features.py`, dan band hilang diimputasi median training saat serving.

Temuan EDA penting: **korelasi fitur–target sangat lemah** (maksimum |r| ≈ 0.20,
itu pun dari `longitude`; semua band spektral < 0.18). Sebabnya `total_carbon_stock`
didominasi **karbon tanah (~90%)** yang tidak terlihat oleh band optik/radar.
Karena itu R² moderat **bukan** akibat data kotor, melainkan keterbatasan sinyal
fitur — bisa dinaikkan dengan menambah fitur relevan (elevasi/pasang-surut,
tekstur, komposit temporal) atau memodelkan komponen karbon secara terpisah.

## Model yang robust (dataset kecil ~300 baris)
Untuk mengurangi overfitting dan memberi estimasi performa yang andal:
- **Regularisasi kuat**: pohon dangkal (`max_depth=4`), `min_child_weight=5`,
  L1/L2 (`reg_alpha`, `reg_lambda`), `gamma`, plus `subsample`/`colsample_bytree`.
- **Cross-validation 5-fold** pada porsi train → estimasi generalisasi
  **mean ± std** (bukan menebak dari satu hold-out 60-baris yang berisik).
  Median imputasi dihitung ulang per-fold (anti-kebocoran).
- **Early stopping** memilih jumlah pohon optimal otomatis; model final dilatih
  ulang pada SELURUH porsi train dengan jumlah pohon hasil CV — booster final
  berisi tepat sejumlah itu, jadi `predict.py` tidak berubah.

Hasil CV & jumlah pohon terpilih dicatat di `metadata.json` pada field `cv`.
Catatan domain: `total_carbon_stock` didominasi **karbon tanah** yang sulit
terlihat dari band spektral, sehingga R² wajar tetap moderat (~0.36, ±0.15
antar-fold) — robustness di sini berarti estimasi jujur & tidak overfit,
bukan angka R² yang dipaksa tinggi.

## Latih
```bash
pip install -r requirements.txt
python train.py --data /path/civ_mangrove_biomass_dataset.csv --out ./artifacts
```

## Artefak model
`train.py` menghasilkan dua format model yang setara (prediksi identik):
- **`artifacts/model.json`** — booster XGBoost native, **portabel antar-versi** (format utama yang dipakai API).
- **`artifacts/model.pkl`** — bundle self-contained (joblib) berisi estimator + metadata; cukup file ini saja untuk inference. Catatan: pickle terikat versi pustaka, jadi `model.json` tetap disarankan untuk portabilitas.

## Inference
```python
from predict import CarbonStockModel

# (a) dari format native (disarankan)
model = CarbonStockModel("./artifacts")

# (b) dari pickle self-contained
model = CarbonStockModel.from_pickle("./artifacts/model.pkl")

preds = model.predict(df_band_mentah)   # mengembalikan stok karbon skala asli
```

## REST API (FastAPI)
Membungkus `predict.py`; memuat artefak sekali saat startup lalu melayani HTTP.
```bash
# 1) latih dulu agar ./artifacts berisi model.json + metadata.json
python train.py --data ./data/civ_mangrove_biomass_dataset.csv --out ./artifacts
# 2) jalankan API
uvicorn app:app --host 0.0.0.0 --port 8000
```
Dokumentasi interaktif (Swagger): <http://localhost:8000/docs>

| Endpoint | Guna |
|----------|------|
| `GET /health`   | health check (200 siap / 503 belum termuat) |
| `GET /metadata` | metrik, CV, split, versi pustaka |
| `POST /predict` | prediksi batch dari daftar record band mentah |

Contoh `POST /predict` (field yang hilang otomatis diimputasi median training):
```json
{ "records": [
  { "longitude": -5.5801, "latitude": 5.1047, "B2": 0.0537, "B3": 0.0730,
    "B4": 0.0607, "B5": 0.1049, "B6": 0.2407, "B7": 0.2916, "B8": 0.2831,
    "B8A": 0.3104, "B11": 0.1370, "B12": 0.0733, "VV": -7.28, "VH": -14.54 },
  { "B8": 0.30, "B4": 0.06 }
] }
```
Respons: `{ "predictions": [822.53, 361.78], "n": 2, "target": "total_carbon_stock" }`

Folder artefak dapat di-override via env `ARTIFACT_DIR` (default `./artifacts`).

## Docker
```bash
# Pastikan ./artifacts sudah ada (hasil train.py) sebelum build.
docker build -t carbon-stock-api .
docker run --rm -p 8000:8000 carbon-stock-api
# Override artefak tanpa rebuild (mount folder lokal):
docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" carbon-stock-api
```
Image berbasis `python:3.13-slim` (menyamai versi training), berjalan sebagai
user non-root, dan punya `HEALTHCHECK` ke `/health`.

## Uji cepat (smoke test)
```bash
python smoke_test.py
```
Mengunduh dataset asli dari Kaggle ke `./data` (bila belum ada), melatih,
menyimpan & memuat ulang artefak, **memverifikasi pembagian 80/20**, lalu
mengevaluasi pada 20% hold-out (metriknya dicek harus cocok dengan metadata)
dan memverifikasi ketiga perbaikan end-to-end. Sumber data:
<https://www.kaggle.com/datasets/salomonkouassi/mangrove-biomass-and-carbon-stocks-dataset>

## Catatan
- `smoke_test.py` kini memakai **dataset asli** (bukan data acak), sehingga
  metrik MAE/RMSE/R² yang dicetak bermakna. Pengunduhan memakai endpoint
  publik Kaggle tanpa kredensial; bila offline, unduh manual dan letakkan
  CSV di `./data`.
