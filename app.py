import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Occupancy Detection",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Occupancy Detection System")
st.write("Prediksi keberadaan orang di dalam ruangan menggunakan Machine Learning")

st.divider()

st.subheader("🔍 Pilih Model")

model_option = st.selectbox(
    "Pilih algoritma yang digunakan:",
    ("Naive Bayes", "Random Forest")
)

# Load model
if model_option == "Naive Bayes":
    model = joblib.load("naive_bayes_model.joblib")
else:
    model = joblib.load("random_forest_model.joblib")

st.success(f"Model **{model_option}** berhasil dimuat")

st.divider()

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
        min_value=0.0,
        value=600.0
    )

    humidity_ratio = st.number_input(
        "📈 Humidity Ratio",
        min_value=0.0,
        value=0.004
    )

st.divider()

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

    st.caption(f"Model yang digunakan: {model_option}")
