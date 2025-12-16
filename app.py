import streamlit as st
import numpy as np
import joblib
import os

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Occupancy Detection",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Occupancy Detection System")
st.write("Prediksi keberadaan orang di dalam ruangan menggunakan Machine Learning")

st.divider()

# ===============================
# FUNGSI LOAD MODEL (AMAN)
# ===============================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan: `{model_path}`")
        st.stop()
    return joblib.load(model_path)

# ===============================
# PILIH MODEL
# ===============================
st.subheader("🔍 Pilih Model")

model_option = st.selectbox(
    "Pilih algoritma yang digunakan:",
    (
        "Naive Bayes",
        "Random Forest (Akurasi ±90%)"
    )
)

if model_option == "Naive Bayes":
    model = load_model("models/naive_bayes_model.joblib")
    model_name = "Naive Bayes"
else:
    model = load_model("models/random_forest_90_model.joblib")
    model_name = "Random Forest (dibatasi, ±90%)"

st.success(f"Model **{model_name}** berhasil dimuat")

st.divider()

# ===============================
# INPUT DATA SENSOR
# ===============================
st.subheader("📊 Input Data Sensor")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=23.0
    )

    humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    light = st.number_input(
        "💡 Light (Lux)",
        min_value=0.0,
        value=300.0
    )

with col2:
    co2 = st.number_input(
        "🧪 CO₂ (ppm)",
        min_value=300.0,
        max_value=2000.0,
        value=600.0
    )

    humidity_ratio = st.number_input(
        "📈 Humidity Ratio",
        min_value=0.001,
        max_value=0.02,
        value=0.004,
        format="%.4f"
    )

st.divider()

# ===============================
# PREDIKSI
# ===============================
if st.button("🔮 Prediksi Occupancy"):
    input_data = np.array([
        temperature,
        humidity,
        light,
        co2,
        humidity_ratio
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    st.subheader("📌 Hasil Prediksi")

    if prediction == 1:
        st.error("🚶‍♂️ **RUANGAN TERDETEKSI ADA ORANG**")
    else:
        st.success("🏠 **RUANGAN TERDETEKSI KOSONG**")

    st.caption(f"Model yang digunakan: {model_name}")
