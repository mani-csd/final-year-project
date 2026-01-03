import streamlit as st
import joblib
import pandas as pd

model, features = joblib.load(r"D:\marketing_campaign_project\models\campaign_model.pkl")

st.title("Marketing Campaign Predictor")

# User inputs
clicks = st.number_input("Clicks")
revenue = st.number_input("Revenue")
roi = st.number_input("ROI")
discount = st.number_input("Discount")

# Create empty input
input_df = pd.DataFrame(0, index=[0], columns=features)

# Fill known values
if "Clicks" in input_df.columns:
    input_df["Clicks"] = clicks
if "Revenue_Generated" in input_df.columns:
    input_df["Revenue_Generated"] = revenue
if "ROI" in input_df.columns:
    input_df["ROI"] = roi
if "Discount_Level" in input_df.columns:
    input_df["Discount_Level"] = discount

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.success("Customer will respond" if pred == 1 else "Customer will NOT respond")
