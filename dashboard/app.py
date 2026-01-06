import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load model and feature list
model, features = joblib.load("models/campaign_model.pkl")

st.set_page_config(page_title="Marketing Campaign Predictor", layout="centered")

# ---------------- HEADER ----------------
st.title("📊 Marketing Campaign Response Predictor")

st.markdown("""
### 🎯 Marketing Campaign Performance & Conversion Analytics  
This system predicts whether a customer will respond to a marketing campaign using machine learning.

**Developed by:**  
C. Harsha Vardhan Balaji Rao  
A. Purushottam  
D. Ganesh  
J. Govardhan  
B. Venkatamani Chandra  

**Department:** B.Tech CSE (Data Science)  
""")

st.info("Model Accuracy: 87.75% (Random Forest Classifier)")

# ---------------- SAMPLE DATA ----------------
if st.button("📌 Load Sample Campaign"):
    st.session_state["clicks"] = 500
    st.session_state["revenue"] = 15000.0
    st.session_state["roi"] = 3.5
    st.session_state["discount"] = 20.0

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    clicks = st.number_input(
        "Clicks",
        min_value=0,
        value=int(st.session_state.get("clicks", 0)),
        help="Total number of customer clicks"
    )

    revenue = st.number_input(
        "Revenue Generated",
        min_value=0.0,
        value=float(st.session_state.get("revenue", 0.0)),
        help="Total revenue from the campaign"
    )

with col2:
    roi = st.number_input(
        "ROI",
        min_value=0.0,
        value=float(st.session_state.get("roi", 0.0)),
        help="Return on Investment"
    )

    discount = st.number_input(
        "Discount Level",
        min_value=0.0,
        value=float(st.session_state.get("discount", 0.0)),
        help="Discount offered (%)"
    )

# ---------------- CREATE INPUT DF ----------------
input_df = pd.DataFrame(0, index=[0], columns=features)

if "Clicks" in input_df.columns:
    input_df["Clicks"] = clicks
if "Revenue_Generated" in input_df.columns:
    input_df["Revenue_Generated"] = revenue
if "ROI" in input_df.columns:
    input_df["ROI"] = roi
if "Discount_Level" in input_df.columns:
    input_df["Discount_Level"] = discount

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict"):
    prob = model.predict_proba(input_df)[0][1]
    pred = 1 if prob > 0.5 else 0

    st.write(f"### 📊 Response Probability: {prob*100:.2f}%")

    if pred == 1:
        st.success("🟢 Customer WILL respond to this campaign")
    else:
        st.error("🔴 Customer will NOT respond to this campaign")

    # -------- FEATURE IMPORTANCE --------
    st.subheader("📌 Top Factors Influencing Customer Response")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance[importance["Importance"] > 0]
    importance = importance.sort_values(by="Importance", ascending=False).head(5)

    fig, ax = plt.subplots()
    ax.barh(importance["Feature"], importance["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 5 Features")
    st.pyplot(fig)

    # -------- PDF REPORT --------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Marketing Campaign Prediction Report", ln=True)
    pdf.cell(0, 10, f"Clicks: {clicks}", ln=True)
    pdf.cell(0, 10, f"Revenue: {revenue}", ln=True)
    pdf.cell(0, 10, f"ROI: {roi}", ln=True)
    pdf.cell(0, 10, f"Discount: {discount}", ln=True)
    pdf.cell(0, 10, f"Response Probability: {prob*100:.2f}%", ln=True)
    pdf.cell(0, 10, "Prediction: " + ("Respond" if pred == 1 else "Not Respond"), ln=True)

    st.download_button(
        label="📄 Download Prediction Report",
        data=pdf.output(dest="S").encode("latin-1"),
        file_name="campaign_prediction.pdf",
        mime="application/pdf"
    )
