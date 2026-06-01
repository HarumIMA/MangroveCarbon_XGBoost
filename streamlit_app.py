"""
streamlit_app.py
================
Antarmuka web ramah-pengguna untuk model prediksi stok karbon mangrove.

Membungkus jalur inference yang sama dengan predict.py (memuat artefak di
./artifacts, menghitung indeks spektral lewat features.py, meng-inverse log1p
ke skala asli). Pengguna non-teknis cukup mengisi nilai band (form sudah terisi
contoh default realistis), menekan tombol, lalu melihat hasil dalam bentuk
metric card.

Model membutuhkan 14 band MENTAH sebagai input:
    longitude, latitude, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, VV, VH
Tiga fitur turunan (NDVI, NDWI, SAVI) dihitung OTOMATIS, jadi pengguna tidak
perlu mengisinya.

Cara menjalankan:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from features import RAW_BANDS, TARGET, add_spectral_indices
from predict import CarbonStockModel

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

# --- Metadata tampilan band (label ramah + bantuan) ------------------------
OPTICAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
SAR_BANDS = ["VV", "VH"]

BAND_LABELS = {
    "longitude": "Bujur (longitude)",
    "latitude": "Lintang (latitude)",
    "B2": "B2 — Biru",
    "B3": "B3 — Hijau",
    "B4": "B4 — Merah",
    "B5": "B5 — Red Edge 1",
    "B6": "B6 — Red Edge 2",
    "B7": "B7 — Red Edge 3",
    "B8": "B8 — NIR",
    "B8A": "B8A — NIR sempit",
    "B11": "B11 — SWIR 1",
    "B12": "B12 — SWIR 2",
    "VV": "VV — radar (dB)",
    "VH": "VH — radar (dB)",
}


# --- Pemuatan sumber daya (di-cache supaya tidak diulang tiap interaksi) ----
@st.cache_resource(show_spinner="Memuat model…")
def load_model() -> CarbonStockModel:
    return CarbonStockModel(str(ARTIFACT_DIR))


@st.cache_data(show_spinner=False)
def load_reference_targets() -> np.ndarray | None:
    """Gabungkan target dari train.csv + test.csv untuk distribusi pembanding."""
    frames = []
    for name in ("train.csv", "test.csv"):
        p = ARTIFACT_DIR / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    ref = pd.concat(frames, ignore_index=True)
    if TARGET not in ref.columns:
        return None
    return ref[TARGET].to_numpy(dtype=float)


def interpret(pred: float, ref: np.ndarray | None) -> tuple[str, str]:
    """Kategori sederhana berdasarkan kuartil data latih (bila tersedia)."""
    if ref is not None and len(ref) >= 4:
        q1, q3 = np.percentile(ref, [25, 75])
        if pred < q1:
            return "Rendah", "Di bawah 25% terendah data latih."
        if pred < q3:
            return "Sedang", "Berada dalam rentang tipikal data latih."
        return "Tinggi", "Di atas 75% data latih — kandidat simpanan karbon besar."
    # Fallback ambang tetap bila referensi tak ada.
    if pred < 100:
        return "Rendah", "Estimasi stok karbon kecil."
    if pred < 800:
        return "Sedang", "Estimasi stok karbon menengah."
    return "Tinggi", "Estimasi stok karbon besar."


# --- Halaman ----------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Stok Karbon Mangrove",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Prediksi Stok Karbon Mangrove")
st.caption(
    "Masukkan nilai band citra satelit, lalu aplikasi memperkirakan "
    "**total stok karbon** pada plot tersebut. Form sudah terisi contoh nilai "
    "realistis — Anda bisa langsung menekan tombol prediksi."
)

# Muat model; bila artefak belum ada, beri instruksi jelas (tanpa crash).
if not (ARTIFACT_DIR / "model.json").exists():
    st.error(
        f"Artefak model tidak ditemukan di `{ARTIFACT_DIR}`.\n\n"
        "Latih model dulu:\n\n"
        "```\npython train.py --data ./data/civ_mangrove_biomass_dataset.csv "
        "--out ./artifacts\n```"
    )
    st.stop()

model = load_model()
meta = model.meta
medians = meta["feature_medians"]          # nilai default contoh (median data latih)
reference = load_reference_targets()

with st.expander("ℹ️ Cara pakai & tentang model", expanded=False):
    st.markdown(
        "- **Band optik (B2–B12):** reflektansi Sentinel-2, biasanya 0–1.\n"
        "- **Band radar (VV, VH):** hamburan balik Sentinel-1 dalam dB (nilai negatif).\n"
        "- **Lokasi (longitude/latitude):** koordinat plot.\n"
        "- Indeks **NDVI, NDWI, SAVI** dihitung otomatis dari band — tidak perlu diisi.\n"
        f"- Model: **{meta.get('model_type', 'XGBoost')}**, target di-transform "
        f"`{meta.get('target_transform')}` lalu dikembalikan ke skala asli."
    )

# --- Form input ------------------------------------------------------------
st.subheader("1️⃣ Masukkan nilai band")

with st.form("input_form"):
    st.markdown("**📍 Lokasi plot**")
    c1, c2 = st.columns(2)
    values: dict[str, float] = {}
    values["longitude"] = c1.number_input(
        BAND_LABELS["longitude"], value=float(medians["longitude"]),
        step=0.01, format="%.4f", help="Koordinat bujur (derajat).")
    values["latitude"] = c2.number_input(
        BAND_LABELS["latitude"], value=float(medians["latitude"]),
        step=0.01, format="%.4f", help="Koordinat lintang (derajat).")

    st.markdown("**🛰️ Band optik Sentinel-2** (reflektansi, umumnya 0–1)")
    opt_cols = st.columns(5)
    for i, b in enumerate(OPTICAL_BANDS):
        values[b] = opt_cols[i % 5].number_input(
            BAND_LABELS[b], value=float(medians[b]),
            min_value=0.0, step=0.001, format="%.4f")

    st.markdown("**📡 Band radar Sentinel-1** (dB, nilai negatif)")
    s1, s2 = st.columns(2)
    values["VV"] = s1.number_input(
        BAND_LABELS["VV"], value=float(medians["VV"]), step=0.1, format="%.2f")
    values["VH"] = s2.number_input(
        BAND_LABELS["VH"], value=float(medians["VH"]), step=0.1, format="%.2f")

    submitted = st.form_submit_button("🔮 Prediksi Stok Karbon", type="primary",
                                      use_container_width=True)

# --- Hasil -----------------------------------------------------------------
if submitted:
    raw = pd.DataFrame([{b: values[b] for b in RAW_BANDS}])
    pred = float(model.predict(raw)[0])
    indices = add_spectral_indices(raw).iloc[0]   # NDVI/NDWI/SAVI dari input
    kategori, penjelasan = interpret(pred, reference)

    st.subheader("2️⃣ Hasil prediksi")

    # Metric card utama + kategori.
    m1, m2 = st.columns([2, 1])
    delta = None
    if reference is not None and len(reference) > 0:
        med = float(np.median(reference))
        delta = f"{pred - med:+,.1f} vs median data latih"
    m1.metric(label="Estimasi Total Carbon Stock", value=f"{pred:,.1f}",
              delta=delta, help="Satuan mengikuti dataset (mis. Mg C per plot).")
    m2.metric(label="Kategori", value=kategori)
    st.info(f"**{kategori}** — {penjelasan}")

    # Metric card indeks spektral turunan (otomatis).
    st.markdown("**Indeks spektral (dihitung otomatis):**")
    i1, i2, i3 = st.columns(3)
    i1.metric("NDVI", f"{indices['NDVI']:.3f}", help="Kerapatan vegetasi (-1..1).")
    i2.metric("NDWI", f"{indices['NDWI']:.3f}", help="Kandungan air (-1..1).")
    i3.metric("SAVI", f"{indices['SAVI']:.3f}", help="Vegetasi terkoreksi tanah.")

    # Visualisasi posisi prediksi pada distribusi data latih.
    if reference is not None and len(reference) > 0:
        st.markdown("**Posisi prediksi terhadap distribusi data latih:**")
        hist_df = pd.DataFrame({"Total carbon stock": reference})
        hist = (
            alt.Chart(hist_df)
            .mark_bar(opacity=0.75, color="#2e7d32")
            .encode(
                x=alt.X("Total carbon stock:Q", bin=alt.Bin(maxbins=30),
                        title="Total carbon stock (data latih)"),
                y=alt.Y("count()", title="Jumlah plot"),
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"Prediksi": [pred]}))
            .mark_rule(color="red", size=3)
            .encode(x="Prediksi:Q",
                    tooltip=[alt.Tooltip("Prediksi:Q", format=",.1f")])
        )
        st.altair_chart(hist + rule, use_container_width=True)
        st.caption("Garis merah = prediksi Anda; batang hijau = sebaran nilai pada data latih.")

# --- Kualitas model (konteks) ----------------------------------------------
st.divider()
st.subheader("📊 Kualitas model")
st.caption("Diukur pada data uji 20% yang tidak ikut dilatih, serta lewat "
           "cross-validation 5-fold (estimasi yang lebih stabil).")
metrics = meta.get("metrics", {})
cv = meta.get("cv", {})
q1, q2, q3, q4 = st.columns(4)
q1.metric("R² (hold-out)", f"{metrics.get('r2', float('nan')):.3f}")
q2.metric("MAE (hold-out)", f"{metrics.get('mae', float('nan')):,.1f}")
q3.metric("RMSE (hold-out)", f"{metrics.get('rmse', float('nan')):,.1f}")
if cv:
    q4.metric("R² (CV 5-fold)", f"{cv.get('r2_mean', float('nan')):.3f}",
              delta=f"± {cv.get('r2_std', 0):.3f}", delta_color="off")

st.caption(
    "Catatan: R² moderat karena total_carbon_stock didominasi karbon tanah "
    "yang sulit dilihat band optik/radar — model dirancang robust (regularisasi "
    "+ CV), bukan dipaksa akurasi tinggi."
)
