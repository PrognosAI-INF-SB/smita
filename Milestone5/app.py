# app.py — Light Theme Predictive Dashboard
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

# ---------------- Configuration ----------------
ROOT = Path(__file__).parent.resolve()
MODEL_PATH = ROOT / "trained_model.h5"
SCALER_PATH = ROOT / "scaler.pkl"
GRAPH_DIR = ROOT / "train_model_graphs"
GRAPH_DIR.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="PrognosAI Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------------- Load Model ----------------
@st.cache_resource
def load_trained_model():
    model = load_model(str(MODEL_PATH), compile=False)
    return model

model = load_trained_model()

# ---------------- Load Scaler ----------------
@st.cache_resource
def load_scaler():
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    else:
        return MinMaxScaler()

scaler = load_scaler()

# ---------------- Helper Functions ----------------
def predict_rul(features):
    scaled = scaler.transform(features)
    X_in = scaled.reshape(scaled.shape[0], 1, scaled.shape[1])
    predictions = model.predict(X_in)
    rul_values = predictions.flatten()
    return rul_values

def status_and_color(rul):
    if rul < 40:
        return "Critical ⚠️", "#FF4D4D"
    elif rul < 70:
        return "Warning 🟠", "#FFA500"
    else:
        return "Healthy ✅", "#00CC66"

def save_rul_graph(machine_id, rul_trend):
    fname = GRAPH_DIR / f"{machine_id}_rul.jpg"
    plt.figure(figsize=(8, 3))
    plt.plot(rul_trend, color="#3b8eed", marker='o')
    plt.title(f"Predicted RUL Trend - {machine_id}")
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname

# ---------------- Sidebar ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/829/829379.png", width=60)
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Input Mode", ["Manual Input", "Upload CSV File"])
feature_count = st.sidebar.number_input("Feature Count", 1, 50, 21)
st.sidebar.info("📘 Ensure your input matches model feature count.")

# ---------------- Title ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #004aad;'>PrognosAI — Predictive Maintenance Dashboard</h1>
    <p style='text-align: center;'>Analyze machine health, predict Remaining Useful Life (RUL), and visualize results.</p>
    """, unsafe_allow_html=True
)

# ---------------- Input Mode 1: Manual ----------------
if mode == "Manual Input":
    st.markdown("### 🔹 Enter Machine Feature Values")
    machine_id = st.text_input("Machine ID", "Machine_1")

    cols = st.columns(7)
    features = []
    for i in range(feature_count):
        with cols[i % 7]:
            val = st.number_input(f"Feature {i+1}", 0.0, 200.0, float(50 + i), key=f"f{i}")
            features.append(val)

    if st.button("🔮 Predict & Generate Dashboard"):
        df = pd.DataFrame([features], columns=[f"f{i+1}" for i in range(feature_count)])
        rul_pred = predict_rul(df.values)[0]
        status, color = status_and_color(rul_pred)

        st.success(f"**Predicted RUL:** {rul_pred:.2f}")
        st.markdown(f"**Machine Status:** <span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

        # Generate trend and graph
        trend = np.linspace(rul_pred, max(rul_pred - 30, 0), 10)
        path = save_rul_graph(machine_id, trend)

        # Subplots Dashboard
        col1, col2, col3 = st.columns(3)

        # Line Chart (RUL Trend)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=trend, mode='lines+markers', line=dict(color='#004aad')))
            fig.update_layout(title="RUL Trend", xaxis_title="Cycle", yaxis_title="RUL", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Pie Chart (Health)
        with col2:
            labels = ["Healthy", "Warning", "Critical"]
            values = [1 if status == "Healthy ✅" else 0,
                      1 if status == "Warning 🟠" else 0,
                      1 if status == "Critical ⚠️" else 0]
            fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
            fig2.update_layout(title="Health Status Distribution", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        # Bar Chart (Feature Summary)
        with col3:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=[f"f{i+1}" for i in range(feature_count)], y=features, marker_color="#3b8eed"))
            fig3.update_layout(title="Feature Values", xaxis_title="Feature", yaxis_title="Value", height=300)
            st.plotly_chart(fig3, use_container_width=True)

        # Download
        with open(path, "rb") as f:
            st.download_button("⬇ Download RUL Graph (JPG)", f, file_name=os.path.basename(path))

# ---------------- Input Mode 2: Upload CSV ----------------
else:
    st.markdown("### 📤 Upload CSV File for Bulk Prediction")
    uploaded = st.file_uploader("Upload your CSV file (each row = one machine)", type=["csv"])

    if uploaded:
        data = pd.read_csv(uploaded)
        st.dataframe(data.head())

        if st.button("🚀 Predict All"):
            rul_values = predict_rul(data.values)
            data["Predicted_RUL"] = rul_values
            statuses, colors = [], []
            for val in rul_values:
                s, c = status_and_color(val)
                statuses.append(s)
                colors.append(c)
            data["Status"] = statuses

            st.success("Predictions Complete ✅")
            st.dataframe(data)

            # Dashboard plots
            col1, col2 = st.columns(2)

            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(x=data.index, y=data["Predicted_RUL"], marker_color="#3b8eed"))
                fig1.update_layout(title="Machine-wise RUL", xaxis_title="Machine Index", yaxis_title="Predicted RUL", height=350)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                counts = data["Status"].value_counts()
                fig2 = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.4)])
                fig2.update_layout(title="Health Distribution", height=350)
                st.plotly_chart(fig2, use_container_width=True)

            st.download_button("⬇ Download Predictions (CSV)", data.to_csv(index=False).encode('utf-8'),
                               file_name="predicted_rul_results.csv", mime="text/csv")