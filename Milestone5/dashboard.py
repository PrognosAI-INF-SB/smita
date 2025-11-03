# Milestone5/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import os
import io

# --- 1. Define Constants and Load Assets ---

# Risk Thresholds (Global)
CRITICAL_LEVEL = 15
WARNING_LEVEL = 30

# Sensor columns and sequence length
SENSOR_COLS = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 
    's13', 's14', 's15', 's17', 's20', 's21'
]
SEQUENCE_LENGTH = 50

@st.cache_resource
def load_assets():
    """Loads the model, scaler, and raw data once."""
    base_dir = os.path.dirname(__file__)
    
    MODEL_PATH = os.path.join(base_dir, '../Milestone2/models/PrognosAI_LSTM.keras')
    SCALER_PATH = os.path.join(base_dir, '../Milestone2/data/processed/scaler.pkl')
    DATA_PATH = os.path.join(base_dir, '../Milestone1/data/train_FD001.txt')
    
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # Column names for all data
        column_names = [
            'unit_nr', 'cycle', 'setting1', 'setting2', 'setting3',
            's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
            's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
            's20', 's21'
        ]
        
        # Load the raw training data
        df = pd.read_csv(DATA_PATH, sep='\s+', header=None, names=column_names)
        
        return model, scaler, df, column_names
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None

# --- 2. Helper Functions ---

def classify_risk(rul_prediction):
    """Classifies a predicted RUL into a risk category."""
    if rul_prediction <= CRITICAL_LEVEL:
        return "🔴 CRITICAL", "Maintain Immediately"
    elif rul_prediction <= WARNING_LEVEL:
        return "🟡 WARNING", "Schedule Maintenance"
    else:
        return "🟢 STABLE", "No Action Required"

def prepare_sequences(engine_df, scaler):
    """Prepares all sequences for a single engine."""
    engine_df_scaled = engine_df.copy()
    engine_df_scaled[SENSOR_COLS] = scaler.transform(engine_df_scaled[SENSOR_COLS])
    
    sequences = []
    for i in range(len(engine_df) - SEQUENCE_LENGTH):
        seq = engine_df_scaled.iloc[i:i+SEQUENCE_LENGTH][SENSOR_COLS].values
        sequences.append(seq)
        
    return np.array(sequences)

def get_last_sequence(df, scaler):
    """Gets the very last sequence from a dataframe."""
    # Create a copy to avoid changing the original dataframe
    df_scaled = df.copy()
    df_scaled[SENSOR_COLS] = scaler.transform(df_scaled[SENSOR_COLS])

    last_sequence_scaled = df_scaled[SENSOR_COLS].tail(SEQUENCE_LENGTH).values
    
    if len(last_sequence_scaled) < SEQUENCE_LENGTH:
        padded_sequence = np.zeros((SEQUENCE_LENGTH, len(SENSOR_COLS)))
        padded_sequence[-len(last_sequence_scaled):] = last_sequence_scaled
        last_sequence_scaled = padded_sequence
        
    return np.expand_dims(last_sequence_scaled, axis=0) # Add batch dim

def create_gauge_chart(rul_value):
    """Creates a Plotly gauge chart for the RUL."""
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = rul_value,
        number = {'suffix': " Cycles", 'font': {'size': 24}},
        title = {'text': "Predicted Remaining Useful Life", 'font': {'size': 20}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': "blue", 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps' : [
                {'range': [0, CRITICAL_LEVEL], 'color': 'rgba(255, 75, 75, 0.7)'}, # Red
                {'range': [CRITICAL_LEVEL, WARNING_LEVEL], 'color': 'rgba(255, 195, 0, 0.7)'}, # Yellow
                {'range': [WARNING_LEVEL, 150], 'color': 'rgba(40, 167, 69, 0.7)'}  # Green
            ],
        }
    ))
    fig_gauge.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': 'white'} if st.session_state.get('theme', 'light') == 'dark' else {'color': 'black'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig_gauge

# --- 3. Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="PrognosAI Dashboard")

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light' 

model, scaler, df_full, col_names = load_assets()

if not all([model, scaler, df_full is not None, col_names]):
    st.error("Failed to load project assets. Please ensure all previous milestones are complete and paths are correct.")
else:
    st.sidebar.title("PrognosAI Navigation")
    nav = st.sidebar.radio("Go to:", [
        "✈️ Fleet Overview", 
        "📄 Batch Prediction (File Upload)", 
        "🔧 Live 'What-If' Simulation" # <-- RENAMED THIS PAGE
    ])

    # Check for theme change
    theme = st.sidebar.selectbox("Select Theme:", ['light', 'dark'])
    st.session_state['theme'] = theme
    if theme == 'dark':
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0E1117;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # ===================================================================
    # PAGE 1: FLEET OVERVIEW
    # ===================================================================
    if nav == "✈️ Fleet Overview":
        st.title("✈️ Fleet Overview Dashboard")

        st.sidebar.header("Engine Selection")
        engine_ids = df_full['unit_nr'].unique()
        selected_engine = st.sidebar.selectbox("Select Engine ID:", engine_ids)

        df_engine = df_full[df_full['unit_nr'] == selected_engine].reset_index(drop=True)

        # 1. Get Current RUL Prediction
        last_sequence_input = get_last_sequence(df_engine, scaler)
        current_rul_prediction = model.predict(last_sequence_input)[0][0]
        risk_level, recommendation = classify_risk(current_rul_prediction)

        # 2. Display Key Metrics
        st.header(f"Engine {selected_engine}: Status Overview")
        col1, col2 = st.columns([2, 1]) # Give gauge more space

        with col1:
            st.plotly_chart(create_gauge_chart(current_rul_prediction), use_container_width=True)

        with col2:
            st.metric("Risk Level", risk_level)
            st.metric("Recommendation", recommendation)
            st.metric("Total Cycles Run", df_engine['cycle'].max())
        st.markdown("---")

        # 3. RUL Forecast Plot & Sensor Data (Code is unchanged)
        # (This section is the same as before...)

        st.header("RUL Degradation Forecast")
        all_sequences = prepare_sequences(df_engine, scaler)
        
        if len(all_sequences) > 0:
            rul_predictions = model.predict(all_sequences).flatten()
            plot_cycles = df_engine['cycle'].iloc[SEQUENCE_LENGTH:].values
            
            fig_rul = go.Figure()
            fig_rul.add_trace(go.Scatter(
                x=plot_cycles, y=rul_predictions, mode='lines', name='Predicted RUL',
                line=dict(color='blue', width=3)
            ))
            fig_rul.add_hrect(y0=0, y1=CRITICAL_LEVEL, line_width=0, fillcolor='rgba(255, 75, 75, 0.2)', name="Critical Zone")
            fig_rul.add_hrect(y0=CRITICAL_LEVEL, y1=WARNING_LEVEL, line_width=0, fillcolor='rgba(255, 195, 0, 0.2)', name="Warning Zone")
            fig_rul.update_layout(title=f"Predicted RUL over Engine {selected_engine}'s Life",
                                  xaxis_title="Cycle", yaxis_title="Predicted RUL", hovermode="x unified")
            st.plotly_chart(fig_rul, use_container_width=True)
        else:
            st.warning("Not enough data to create a RUL forecast for this engine.")

        st.header("Live Sensor Data")
        fig_sensors = go.Figure()
        for sensor in SENSOR_COLS:
            fig_sensors.add_trace(go.Scatter(
                x=df_engine['cycle'], y=df_engine[sensor], mode='lines', name=sensor
            ))
        fig_sensors.update_layout(title="Sensor Data over Time", xaxis_title="Cycle",
                                  yaxis_title="Sensor Value", hovermode="x unified")
        st.plotly_chart(fig_sensors, use_container_width=True)

    # ===================================================================
    # PAGE 2: BATCH PREDICTION (File Upload)
    # ===================================================================
    elif nav == "📄 Batch Prediction (File Upload)":
        st.title("📄 Batch Prediction from New File")
        st.info("Upload a file in the same format as the NASA text files (e.g., `test_FD001.txt`).")
        
        uploaded_file = st.file_uploader("Choose a file (e.g., test_FD001.txt)", type=["txt", "csv"])

        if uploaded_file is not None:
            try:
                df_test_raw = pd.read_csv(uploaded_file, sep='\s+', header=None, names=col_names)
                st.success("File uploaded and read successfully!")
                
                # Process the data
                # (This section is the same as before...)
                
                test_sequences = []
                engine_ids_list = []
                engine_ids = df_test_raw['unit_nr'].unique()

                for engine_id in engine_ids:
                    engine_df = df_test_raw[df_test_raw['unit_nr'] == engine_id]
                    last_sequence_input = get_last_sequence(engine_df, scaler)
                    test_sequences.append(last_sequence_input[0])
                    engine_ids_list.append(engine_id)

                X_test = np.array(test_sequences)

                y_pred = model.predict(X_test).flatten()
                
                results = pd.DataFrame({
                    'Engine_ID': engine_ids_list,
                    'Predicted_RUL': np.round(y_pred, 2)
                })
                results['Risk_Level'] = results['Predicted_RUL'].apply(lambda x: classify_risk(x)[0])
                results = results.sort_values(by='Predicted_RUL')
                
                st.header("Maintenance Alert Report")
                st.dataframe(results, use_container_width=True)
                
                critical_count = (results['Risk_Level'] == "🔴 CRITICAL").sum()
                warning_count = (results['Risk_Level'] == "🟡 WARNING").sum()
                stable_count = len(results) - critical_count - warning_count
                
                st.header("Alert Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 CRITICAL Engines", critical_count)
                col2.metric("🟡 WARNING Engines", warning_count)
                col3.metric("🟢 STABLE Engines", stable_count)
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.error("Please ensure the file is a space-separated CSV/TXT with 26 columns and no header.")


    # ===================================================================
    # PAGE 3: LIVE "WHAT-IF" SIMULATION (Your new feature)
    # ===================================================================
    elif nav == "🔧 Live 'What-If' Simulation":
        st.title("🔧 Live 'What-If' Simulation")
        st.info(
            """
            Select a **Base Engine** to load its history. 
            Then, modify its latest sensor values to see how they affect the RUL.
            """
        )
        
        # --- 1. Select Base Engine ---
        engine_ids = df_full['unit_nr'].unique()
        base_engine_id = st.selectbox("Select Base Engine for History:", engine_ids, key="sim_engine")
        
        df_engine = df_full[df_full['unit_nr'] == base_engine_id].reset_index(drop=True)
        last_known_row = df_engine.iloc[-1]
        
        st.markdown(f"**Loaded History for Engine {base_engine_id}**. Last cycle: {last_known_row['cycle']}")
        
        # --- 2. Create Input Boxes for 14 Features ---
        st.subheader("Modify Latest Sensor Values:")
        
        # Create 4 columns for a cleaner layout
        cols = st.columns(4)
        new_sensor_values = {}
        
        for i, sensor_name in enumerate(SENSOR_COLS):
            with cols[i % 4]:
                # Pre-fill the box with the last known value for that sensor
                last_value = float(last_known_row[sensor_name])
                
                new_val = st.number_input(
                    label=sensor_name,
                    value=last_value,
                    format="%.4f" # Show 4 decimal places
                )
                new_sensor_values[sensor_name] = new_val

        # --- 3. Run Simulation ---
        if st.button("Run Simulation", type="primary"):
            
            # 1. Create a new row for the simulated data
            simulated_row = last_known_row.copy()
            simulated_row['cycle'] += 1 # Increment the cycle
            
            # 2. Update the row with the new sensor values from the user
            for sensor, value in new_sensor_values.items():
                simulated_row[sensor] = value
                
            # 3. Append this new row to the engine's history
            simulated_df = pd.concat([df_engine, simulated_row.to_frame().T], ignore_index=True)
            
            # 4. Get the last sequence from this *new* dataframe
            last_sequence_input = get_last_sequence(simulated_df, scaler)
            
            # 5. Make the prediction
            simulated_rul = model.predict(last_sequence_input)[0][0]
            
            st.header(f"Simulation Result for Engine {base_engine_id} (Cycle {simulated_row['cycle']})")
            
            risk_level, recommendation = classify_risk(simulated_rul)
            
            # 6. Display the new gauge and metrics
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(create_gauge_chart(simulated_rul), use_container_width=True)
            with col2:
                st.metric("Risk Level", risk_level)
                st.metric("Recommendation", recommendation)
                
            # Show the original vs. new RUL
            original_rul = model.predict(get_last_sequence(df_engine, scaler))[0][0]
            st.metric("Original RUL", f"{original_rul:.1f} Cycles")
            st.metric("New Simulated RUL", f"{simulated_rul:.1f} Cycles")