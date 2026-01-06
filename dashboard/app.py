import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and feature list
model, features = joblib.load("models/campaign_model.pkl")

st.set_page_config(page_title="Marketing Campaign Predictor", layout="centered")

st.title("📊 Marketing Campaign Response Predictor")
st.write("Enter campaign details to predict whether customers will respond.")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    clicks = st.number_input("Clicks", min_value=0, help="Total number of customer clicks")
    revenue = st.number_input("Revenue Generated", min_value=0.0, help="Total revenue from campaign")

with col2:
    roi = st.number_input("ROI", min_value=0.0, help="Return on Investment")
    discount = st.number_input("Discount Level", min_value=0.0, help="Discount percentage offered")

# Create empty input dataframe
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

# Prediction
if st.button("🔮 Predict"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("🟢 Customer WILL respond to this campaign")
    else:
        st.error("🔴 Customer will NOT respond to this campaign")

# Feature Importance
st.subheader("📌 Top Factors Influencing Customer Response")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(5)

fig, ax = plt.subplots()
ax.barh(importance["Feature"], importance["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.set_title("Top 5 Features")

st.pyplot(fig)
