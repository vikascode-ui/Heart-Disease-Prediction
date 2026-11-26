"""Streamlit interface for the educational heart disease ML model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.caption("Educational machine-learning demonstration — not a medical diagnostic tool.")

if not MODEL_PATH.exists():
    st.warning("Model file not found. First run `python train.py` to train and save the model.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
st.info(f"Model loaded: {bundle.get('model_name', 'ML classifier')}")

with st.form("prediction_form"):
    st.subheader("Enter the dataset features")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=54)
        sex = st.selectbox("Sex (dataset code)", [0, 1], format_func=lambda x: "0 — Female" if x == 0 else "1 — Male")
        cp = st.selectbox("Chest pain type (cp)", [1, 2, 3, 4])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=70, max_value=250, value=130)
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=80, max_value=700, value=240)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    with c2:
        thalach = st.number_input("Maximum heart rate (thalach)", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", [1, 2, 3])
        ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thal dataset code", [3, 6, 7])
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    st.subheader("Model output")
    st.metric("Predicted class", "Disease present in dataset label" if prediction else "No disease in dataset label")
    st.progress(probability)
    st.write(f"Model score for positive class: **{probability:.1%}**")
    st.warning("This is a statistical prediction from a small historical dataset. It cannot confirm or rule out heart disease. Consult a qualified healthcare professional for medical concerns.")
