import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import defaultdict
from sklearn.linear_model import LogisticRegression # Import required model classes for type checking (if needed)
from sklearn.svm import SVC

# ==============================================================================
# --- MOCK ML ARTIFACTS (Since actual .joblib files cannot be provided) ---
# NOTE: Replace the mocks with actual joblib loading in your real environment.
# ==============================================================================

# Define the expected feature set that the trained model requires
# This must match the columns fed to the scaler during training
EXPECTED_FEATURES = [
    "Model_encoded",
    "Color_encoded",
    "Engine_Type_encoded",
    "Year"
]

class MockLabelEncoder:
    """Mock for the fitted target LabelEncoder and feature encoders."""
    def __init__(self, known_mappings):
        # known_mappings structure: {"FeatureName": {"Label1": 0, "Label2": 1, ...}}
        self.known_mappings = known_mappings

    def transform_single(self, feature_name, value):
        """Returns the encoded integer, or -1 if the value is unseen."""
        # Use .get() method on the mapping dictionary for safe lookup
        return self.known_mappings.get(feature_name, {}).get(value, -1)

    def inverse_transform(self, encoded_value):
        """Simulates inverse transform of the model's output (0 or 1)."""
        # Assumes the target variable was encoded into two classes: 0 and 1
        class_map = {0: "Low Sales Potential", 1: "High Sales Potential"}
        # Map the model's prediction array (e.g., [0]) to the text label
        return [class_map.get(v, "Unknown") for v in encoded_value]


# Mock known mappings for demonstration (must reflect the training data)
MOCK_MAPPINGS = {
    "Model": {"3 Series": 0, "5 Series": 1, "X5": 2, "1 Series": 3, "Z4": 4},
    "Color": {"Black": 0, "White": 1, "Blue": 2, "Red": 3},
    "Engine_Type": {"Diesel": 0, "Petrol": 1, "Electric": 2, "Hybrid": 3}
}

class MockScaler:
    """Simulates a StandardScaler. In a real app, this holds mean/std values."""
    def __init__(self, feature_names_in):
        self.feature_names_in = feature_names_in

    def transform(self, df):
        """Simulates scaling by converting the DataFrame to a NumPy array."""
        # NOTE: The actual scaler would apply mean subtraction and division by std dev here.
        return df[self.feature_names_in].values

class MockModel:
    """Simulates the trained Logistic Regression or SVC model."""
    def predict(self, scaled_input):
        """Returns a dummy prediction (0 or 1)."""
        # A simple mock: always predict the 'Low Sales Potential' (encoded as 0)
        return np.array([0])

# ==============================================================================
# --- ARTIFACT LOADING & INITIALIZATION (using MOCKS) ---
# ==============================================================================

# Update the MODEL_PATH based on the final decision from the notebook
MODEL_PATH = "models/best_final_model.joblib" 
SCALER_PATH = "models/scaler.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

@st.cache_resource
def load_artifacts():
    """Loads all ML artifacts and handles potential loading errors."""
    try:
        # --- UNCOMMENT FOR REAL USE ---
        # model = joblib.load(MODEL_PATH)
        # scaler = joblib.load(SCALER_PATH)
        # label_encoder = joblib.load(ENCODER_PATH)

        # --- MOCK SETUP ---
        model = MockModel()
        # Ensure the MockScaler is initialized with the required feature names
        scaler = MockScaler(EXPECTED_FEATURES) 
        # Use the mock mappings for the LabelEncoder
        label_encoder = MockLabelEncoder(MOCK_MAPPINGS) 
        
        return model, scaler, label_encoder, True

    except FileNotFoundError:
        st.sidebar.warning(f"ML Artifacts not found. Using Mock Models. Please run the training notebook and ensure files are saved to the 'models' directory.")
        return MockModel(), MockScaler(EXPECTED_FEATURES), MockLabelEncoder(MOCK_MAPPINGS), False
    except Exception as e:
        st.sidebar.error(f"Error loading ML artifacts: {e}")
        return None, None, None, False

model, scaler, label_encoder, artifacts_loaded = load_artifacts()

if model is None:
    st.error("Application failed to initialize.")
    st.stop()


# ==============================================================================
# --- STREAMLIT UI ---
# ==============================================================================

st.set_page_config(page_title="BMW Sales Predictor", layout="wide")

st.markdown("""
    <style>
        .stButton>button {
            background-color: #0070c0;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .prediction-box {
            background-color: #f0f8ff;
            border-left: 5px solid #0070c0;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🚗 BMW Sales Classification Predictor")
st.markdown("Predict the sales potential (High or Low) for a given BMW vehicle configuration using a Linear or SVM Model.")


# ==============================================================================
# --- USER INPUTS ---
# ==============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Characteristics")
    
    # Input for Categorical Features (Using known keys for better UX)
    # Extract keys from MOCK_MAPPINGS for dropdown suggestions
    
    model_options = list(MOCK_MAPPINGS["Model"].keys())
    model_name = st.selectbox(
        "Model Name",
        options=model_options,
        index=0
    )

    color_options = list(MOCK_MAPPINGS["Color"].keys())
    color = st.selectbox(
        "Color",
        options=color_options,
        index=0
    )

with col2:
    st.subheader("Engine & Age")
    
    engine_type_options = list(MOCK_MAPPINGS["Engine_Type"].keys())
    engine_type = st.selectbox(
        "Engine Type",
        options=engine_type_options,
        index=0
    )
    
    # Input for Numerical Feature
    year = st.number_input(
        "Manufacturing Year", 
        min_value=1990, 
        max_value=2030, 
        value=2018, 
        step=1
    )

# ==============================================================================
# --- INPUT PROCESSING LOGIC ---
# ==============================================================================

def build_input(year, model_name, color, engine_type):
    """
    Creates a DataFrame with all necessary features, safely encodes
    categoricals, and reorders columns for the scaler/model.
    """
    # 1. Start with the input data (using raw column names)
    user_data = {
        "Model": model_name,
        "Color": color,
        "Engine_Type": engine_type,
        "Year": year,
    }

    df = pd.DataFrame([user_data])

    # 2. Encode Categoricals SAFELY
    encoded_values = {}
    unseen_flag = False

    for raw_col, encoded_col in [("Model", "Model_encoded"), ("Color", "Color_encoded"), ("Engine_Type", "Engine_Type_encoded")]:
        encoded_val = label_encoder.transform_single(raw_col, df[raw_col].iloc[0])
        encoded_values[encoded_col] = encoded_val
        if encoded_val == -1:
            unseen_flag = True

    # 3. Combine encoded features and the numerical feature
    input_features = {
        "Model_encoded": [encoded_values["Model_encoded"]],
        "Color_encoded": [encoded_values["Color_encoded"]],
        "Engine_Type_encoded": [encoded_values["Engine_Type_encoded"]],
        "Year": [year]
    }
    
    input_df = pd.DataFrame(input_features)

    # 4. Reorder columns exactly as scaler expects (CRITICAL STEP)
    input_df = input_df[EXPECTED_FEATURES]

    return input_df, unseen_flag


# ==============================================================================
# --- PREDICTION EXECUTION ---
# ==============================================================================

if st.button("Predict Sales Potential"):
    
    # 1. Build & Prepare Input DataFrame
    input_df, unseen_flag = build_input(year, model_name, color, engine_type)

    try:
        # 2. Scale
        # Input_df contains only numerical data in the correct order
        scaled_input = scaler.transform(input_df)

        # 3. Predict (returns encoded label, e.g., [0] or [1])
        pred_encoded = model.predict(scaled_input)

        # 4. Inverse Transform (converts encoded label to human-readable string)
        pred_label = label_encoder.inverse_transform(pred_encoded)

        st.markdown(f"""
            <div class="prediction-box">
                ### 🎯 Prediction Result
                <p style='font-size: 24px; font-weight: bold;'>
                Sales Potential: {pred_label[0]}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if not artifacts_loaded:
            st.info("⚠️ Note: Currently running with **Mock Data** and a Mock Model. Results are not real.")

        # Display how unseen features are handled
        if unseen_flag:
            st.warning("⚠️ Warning: One or more categorical inputs (Model, Color, or Engine Type) were **unseen** during the original model training and were encoded as -1. This may affect prediction accuracy.")
        
        # Optional: Show the processed data for transparency
        with st.expander("Show Processed Features"):
             st.dataframe(input_df)
             st.write("Scaled Input for Model:", scaled_input)


    except Exception as e:
        st.error(f"❌ An internal error occurred during prediction: {str(e)}")
        st.write("Please check the console for details if the ML artifacts are real.")