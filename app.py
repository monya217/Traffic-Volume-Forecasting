import streamlit as st
import pandas as pd
import joblib
import os

FEATURE_COLS = [
    "hour", "day", "month", "weekday", "is_weekend",
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "lag1", "lag2"
]

# Show current working directory for debugging
st.write("Current working directory:", os.getcwd())

@st.cache_resource
def load_model():
    try:
        return joblib.load(r"D:\College\Semester 7\RAI\lab5\models\traffic_rf_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("🚦 Smart City: Traffic Volume Forecasting")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        pred = model.predict(df[FEATURE_COLS])
        df["Predicted Traffic Volume"] = pred

        st.subheader("Predictions")
        st.dataframe(df)

        st.line_chart(df["Predicted Traffic Volume"])

else:
    st.info("Upload a CSV to get predictions.")
