import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import io
from datetime import datetime

# plotting
import plotly.express as px
import plotly.graph_objects as go

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# sklearn scaler fallback
from sklearn.preprocessing import MinMaxScaler

# ---------- CONFIG ----------
MODEL_PATHS_TO_TRY = [
    r"D:\PrognosAI_Project\Milestone5\trained_model.h5",
    r"D:\PrognosAI_Project\Milestone3\trained_model.h5",
    "Milestone5/trained_model.h5",
    "Milestone3/trained_model.h5"
]

SCALER_PATHS_TO_TRY = [
    r"D:\PrognosAI_Project\Milestone3\scaler.pkl",
    r"D:\PrognosAI_Project\Milestone5\scaler.pkl",
    "Milestone3/scaler.pkl",
    "Milestone5/scaler.pkl"
]

DEFAULT_NUM_FEATURES = 21  # you confirmed 21 features

st.set_page_config(page_title="PrognosAI — Predictive Maintenance Dashboard",
                   layout="wide")

# ---------- Helpers ----------
def find_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return None

@st.cache_resource
def load_keras_model(model_path):
    # load without compile to avoid metric deserialization errors
    model = load_model(model_path, compile=False)
    return model

@st.cache_resource
def load_scaler_or_none(path):
    if not path:
        return None
    try:
        scaler = joblib.load(path)
        return scaler
    except Exception:
        return None

def prepare_input_for_model(raw_array, expected_features=DEFAULT_NUM_FEATURES):
    """raw_array: shape (n_features,) or (1,n_features) -> returns (1,1,features)"""
    arr = np.array(raw_array).reshape(1, expected_features)
    arr = arr.astype(np.float32)
    arr = arr.reshape(1, 1, expected_features)  # model expects (batch, timesteps, features)
    return arr

def health_label_from_rul(rul_val):
    if rul_val < 40:
        return "Critical", "red"
    elif rul_val < 70:
        return "Warning", "orange"
    else:
        return "Healthy", "green"

# ---------- Load model & scaler ----------
st.sidebar.header("Model / Data")
model_path = find_existing(MODEL_PATHS_TO_TRY)
scaler_path = find_existing(SCALER_PATHS_TO_TRY)

if model_path:
    st.sidebar.success(f"Model found: {os.path.basename(model_path)}")
    model = load_keras_model(model_path)
else:
    st.sidebar.error("Model not found. Put trained_model.h5 in Milestone5 or Milestone3.")
    st.stop()

if scaler_path:
    scaler = load_scaler_or_none(scaler_path)
    if scaler is not None:
        st.sidebar.success(f"Scaler loaded: {os.path.basename(scaler_path)}")
    else:
        st.sidebar.warning("Scaler file exists but couldn't be loaded; continuing without scaler.")
else:
    scaler = None
    st.sidebar.info("No scaler found — model inputs will be used as-is (or optionally scaled in-app).")

# ---------- UI: Inputs ----------
st.title("🚀 PrognosAI — Predictive Maintenance Dashboard")
st.markdown("Enter sensor / feature values and click **Predict**. The app will display predicted RUL and interactive charts.")

with st.sidebar.expander("Input Options", expanded=True):
    st.write("Provide values for the engine / machine sensors (21 features).")
    input_values = []
    # organize into 3 columns in the sidebar for compactness
    cols = st.columns(1)
    # create 21 fields grouped as columns
    for i in range(1, DEFAULT_NUM_FEATURES + 1):
        default = 0.0
        v = st.number_input(f"feature{i}", value=float(default), step=0.1, format="%.4f", key=f"f{i}")
        input_values.append(v)

    st.write("---")
    custom_scale = st.checkbox("Force MinMax-scale inputs (0-1) using manual max values", value=False)
    if custom_scale:
        st.markdown("If you don't have `scaler.pkl`, you can set a manual max to scale features by dividing.")
        manual_max = st.number_input("Manual max (used for dividing all features)", value=100.0, step=1.0)

    predict_btn = st.button("🔮 Predict RUL & Update Dashboard")

# ---------- Prepare DataFrame for display / download ----------
input_df = pd.DataFrame([input_values], columns=[f"feature{i}" for i in range(1, DEFAULT_NUM_FEATURES + 1)])

# ---------- Main: Predict & Visualize ----------
if predict_btn:
    try:
        # Scale inputs if scaler available or manual scaling requested
        if scaler is not None:
            # Many scalers were fitted with different number of features — ensure dims match
            try:
                scaled = scaler.transform(input_df.values)  # scaler expects 2D
            except Exception as e:
                st.warning("Scaler transform failed (feature mismatch). Using raw inputs.")
                scaled = input_df.values.astype(np.float32)
        else:
            if custom_scale:
                scaled = (input_df.values / float(manual_max)).astype(np.float32)
            else:
                scaled = input_df.values.astype(np.float32)

        # prepare shape (1,1,features)
        model_input = scaled.reshape(1, 1, DEFAULT_NUM_FEATURES)

        # predict
        pred = model.predict(model_input)
        # assume model returns RUL in first output
        predicted_rul = float(pred.flatten()[0])
        predicted_rul_rounded = round(predicted_rul, 2)

        # Determine status
        status_text, status_color = health_label_from_rul(predicted_rul)

        # Show top cards
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted RUL", f"{predicted_rul_rounded}")
            st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold'>{status_text}</span>", unsafe_allow_html=True)
            st.write("Model path:", os.path.basename(model_path))
            if scaler_path:
                st.write("Scaler:", os.path.basename(scaler_path))
        with col2:
            st.write("Input Features (preview)")
            st.dataframe(input_df.T.rename(columns={0: "value"}), height=240)

        # Build interactive charts using Plotly
        # 1) RUL trend (simulate small trend around predicted value)
        cycles = np.arange(1, 11)
        simulated_trend = np.clip(predicted_rul + np.random.normal(0, predicted_rul*0.05, size=len(cycles)), a_min=0, a_max=None)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=cycles, y=simulated_trend, mode='lines+markers', name='Predicted RUL'))
        fig_trend.update_layout(title="Interactive RUL Trend", xaxis_title="Cycle", yaxis_title="RUL", template="plotly_white", height=350)

        # 2) Feature bar chart (absolute values or scaled values)
        feature_vals = input_df.values.flatten()
        feat_names = [f"f{i}" for i in range(1, DEFAULT_NUM_FEATURES+1)]
        df_feats = pd.DataFrame({"feature": feat_names, "value": feature_vals})
        fig_bar = px.bar(df_feats, x="feature", y="value", title="Feature Values", height=350)

        # 3) Pie chart for health segmentation (simple)
        # We'll create three buckets using the single predicted RUL:
        pie_df = pd.DataFrame({"status": [status_text, "Other"], "count": [1, 0]})
        fig_pie = px.pie(pie_df, values="count", names="status", title="Health Status (current sample)")

        # Layout charts
        st.plotly_chart(fig_trend, use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(fig_bar, use_container_width=True)
        right.plotly_chart(fig_pie, use_container_width=True)

        # Save result for download
        results = {
            "predicted_rul": [predicted_rul_rounded],
            "status": [status_text],
            "model_used": [os.path.basename(model_path)],
            "timestamp": [datetime.now().isoformat()]
        }
        results_df = pd.DataFrame(results).join(input_df)

        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        st.download_button("⬇ Download Prediction CSV", data=csv_bytes, file_name="rul_prediction.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)

else:
    # show demo placeholders until user clicks predict
    st.info("Fill the 21 features on the left and click **Predict RUL & Update Dashboard** to see results.")
    # show empty example charts
    sample_x = np.arange(1, 11)
    sample_y = np.linspace(100, 60, 10)

    fig_demo = go.Figure([go.Scatter(x=sample_x, y=sample_y, mode="lines+markers")])
    fig_demo.update_layout(title="Example RUL Trend (demo)", template="plotly_white", height=300)
    st.plotly_chart(fig_demo, use_container_width=True)

    # feature histogram demo
    demo_df = pd.DataFrame({"feature": [f"f{i}" for i in range(1, DEFAULT_NUM_FEATURES+1)],
                            "value": np.random.uniform(0, 1, DEFAULT_NUM_FEATURES)})
    st.plotly_chart(px.bar(demo_df, x="feature", y="value", title="Feature values (demo)"), use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("Built with using Streamlit — enter feature values and predict machine health (RUL).")
