"""Small Streamlit UI for the California Housing model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("california_housing_model.pkl")

st.set_page_config(page_title="California Housing Predictor", page_icon="🏠")
st.title("California Housing Price Prediction")
st.write(
    "Enter district-level characteristics to estimate median house value. "
    "This is a learning demo, not a valuation tool for an individual property."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    st.subheader("Input features")
    med_inc = st.number_input("Median income (MedInc)", min_value=0.0, value=3.5, step=0.1,
                              help="Dataset-specific median income measure.")
    house_age = st.number_input("Median house age (HouseAge)", min_value=1.0, value=25.0, step=1.0)
    ave_rooms = st.number_input("Average rooms (AveRooms)", min_value=0.0, value=5.0, step=0.1)
    ave_bedrms = st.number_input("Average bedrooms (AveBedrms)", min_value=0.0, value=1.0, step=0.1)
    population = st.number_input("Population", min_value=0.0, value=1000.0, step=50.0)
    ave_occup = st.number_input("Average occupancy (AveOccup)", min_value=0.0, value=3.0, step=0.1)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.0, step=0.1)
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-119.0, step=0.1)

    submitted = st.form_submit_button("Predict median house value")

if submitted:
    input_df = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = float(model.predict(input_df)[0])
    dollars = prediction * 100_000

    st.success(f"Predicted median value: ${dollars:,.0f}")
    st.caption(
        f"Model output: {prediction:.3f} in units of $100,000. "
        "Dataset values are capped and represent district-level data."
    )
