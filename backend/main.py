from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import shap
import warnings
import os
import numpy as np

# Suppress a known SHAP warning related to multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the FastAPI application.
app = FastAPI()

# Add CORS middleware to allow requests from the React frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A dictionary to store our pre-trained ensemble models.
models = {}
# A dictionary to store the SHAP TreeExplainer objects.
explainers = {}
# A dictionary to store the original datasets used for SHAP.
datasets = {}
# A dictionary to store the mean values of each feature for imputation.
feature_means = {}
# A dictionary to store the feature names for each model.
feature_names_dict = {}
# A dictionary to map disease names to their target column.
target_columns = {
    'heart': 'HeartDisease',
    'diabetes': 'Outcome',
    'kidney': 'Chronic Kidney Disease: yes',
    'hypertension': 'Has_Hypertension_Yes',
}

# Map simple disease names to the exact model filenames.
model_filenames = {
    'kidney': 'kidney_disease_prediction_ensemble_model.joblib',
    'hypertension': 'hypertension_prediction_ensemble_model.joblib',
    'heart': 'heart_disease_prediction_ensemble_model.joblib',
    'diabetes': 'diabetes_prediction_ensemble_model.joblib',
}

# Load the models and data on application startup.
try:
    model_folder = 'models'
    for disease in ['heart', 'diabetes', 'kidney', 'hypertension']:
        model_filename = model_filenames[disease]
        model_path = os.path.join(model_folder, model_filename)
        
        models[disease] = joblib.load(model_path)
        
        dataset_path = os.path.join('data', f'{disease}_cleaned.csv')
        df = pd.read_csv(dataset_path)

        df = pd.get_dummies(df, drop_first=True)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(col, errors='coerce')
            elif df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
        
        # Define feature names consistently for each disease
        feature_names = df.drop(target_columns[disease], axis=1, errors='ignore').columns.tolist()
        feature_names_dict[disease] = feature_names

        feature_means[disease] = df.drop(target_columns[disease], axis=1, errors='ignore').mean()
        
        df.fillna(df.mean(), inplace=True)
        datasets[disease] = df

    for disease in models:
        rf_model = models[disease].estimators_[0]
        
        # Initialize SHAP explainer without the 'data' parameter to avoid potential alignment issues
        explainers[disease] = shap.TreeExplainer(rf_model)
        
    print("All models, datasets, and SHAP explainers loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure the `models/` and `data/` directories exist and contain the required files.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during startup: {e}")
    exit(1)


# Pydantic models to validate the input data for each disease.
class HeartInput(BaseModel):
    Age: float
    Sex: float
    RestingBP: float
    Cholesterol: float
    FastingBS: float
    MaxHR: float
    ExerciseAngina: float
    Oldpeak: float
    ChestPainType_ATA: float
    ChestPainType_NAP: float
    ChestPainType_TA: float
    RestingECG_Normal: float
    RestingECG_ST: float
    ST_Slope_Flat: float
    ST_Slope_Up: float

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class KidneyInput(BaseModel):
    Age_yrs: float
    Blood_Pressure_mm_Hg: float
    Specific_Gravity: float
    Albumin: float
    Sugar: float
    Blood_Glucose_Random_mgs_dL: float
    Blood_Urea_mgs_dL: float
    Serum_Creatinine_mgs_dL: float
    Sodium_mEq_L: float
    Potassium_mEq_L: float
    Hemoglobin_gms: float
    Packed_Cell_Volume: float
    White_Blood_Cells_cells_cmm: float
    Red_Blood_Cells_millions_cmm: float
    Red_Blood_Cells_normal: float
    Pus_Cells_normal: float
    Pus_Cell_Clumps_present: float
    Bacteria_present: float
    Hypertension_yes: float
    Diabetes_Mellitus_yes: float
    Coronary_Artery_Disease_yes: float
    Appetite_poor: float
    Pedal_Edema_yes: float
    Anemia_yes: float

class HypertensionInput(BaseModel):
    Age: float
    Salt_Intake: float
    Stress_Score: float
    Sleep_Duration: float
    BMI: float
    BP_History_Normal: float
    BP_History_Prehypertension: float
    Medication_Beta_Blocker: float
    Medication_Diuretic: float
    Medication_Other: float
    Exercise_Level_Low: float
    Exercise_Level_Moderate: float
    Smoking_Status_Smoker: float
    Family_History_Yes: float


def get_input_model(disease: str):
    """Helper function to get the correct Pydantic input model."""
    if disease == 'heart':
        return HeartInput
    elif disease == 'diabetes':
        return DiabetesInput
    elif disease == 'kidney':
        return KidneyInput
    elif disease == 'hypertension':
        return HypertensionInput
    raise HTTPException(status_code=404, detail="Disease model not found")


@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Medical Predictor API is running!"}


@app.post("/predict/{disease_name}")
def predict(disease_name: str, data: Dict[str, Any]):
    """
    Endpoint to make a prediction for a given disease.
    """
    model = models.get(disease_name)
    feature_names = feature_names_dict.get(disease_name)
    if model is None or feature_names is None:
        raise HTTPException(status_code=404, detail="Model or feature names for this disease not found.")

    try:
        # Create a DataFrame from the incoming data
        input_df = pd.DataFrame([data])
        
        # Re-index the DataFrame to match the training data's column order.
        input_df = input_df.reindex(columns=feature_names, fill_value=np.nan)

        # Clean data and fill NaNs with pre-calculated means
        input_df.replace('', np.nan, inplace=True)
        input_df.fillna(feature_means[disease_name], inplace=True)
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "probability_has_disease": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain/{disease_name}")
def explain(disease_name: str, data: Dict[str, Any]):
    """
    Endpoint to provide a SHAP explanation for a given prediction.
    """
    explainer = explainers.get(disease_name)
    feature_names = feature_names_dict.get(disease_name)
    if explainer is None or feature_names is None:
        raise HTTPException(status_code=404, detail="Explainer or feature names for this disease not found.")

    try:
        input_df = pd.DataFrame([data])
        
        # This is the crucial step to prevent the mismatch.
        # We explicitly reindex the input DataFrame to match the feature names list.
        input_df = input_df.reindex(columns=feature_names, fill_value=np.nan)
        input_df.replace('', np.nan, inplace=True)
        input_df.fillna(feature_means[disease_name], inplace=True)
        
        # Get the raw SHAP values from the explainer.
        shap_values_raw = explainer.shap_values(input_df)

        # Handle the different output formats of shap_values.
        # Based on the debug output, we know it's a single numpy array with shape (1, 15, 2)
        if isinstance(shap_values_raw, list):
            # Fallback for when SHAP returns a list of two arrays.
            if len(shap_values_raw) == 2:
                shap_values = shap_values_raw[1]
            else:
                raise ValueError("SHAP explainer returned a list of an unexpected length.")
        elif isinstance(shap_values_raw, np.ndarray):
            # The core fix for the (1, 15, 2) shape.
            if shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == 2:
                # Select the SHAP values for the positive class (index 1)
                shap_values = shap_values_raw[0, :, 1]
            else:
                # For a single-class model or regression, shap_values is a single array.
                shap_values = shap_values_raw
        else:
            raise ValueError("SHAP values format is not as expected.")
        
        # Flatten the selected array into a 1D vector.
        shap_values = shap_values.flatten()

        # FINAL check to ensure no length mismatch.
        if len(feature_names) != len(shap_values):
            raise ValueError(
                f"Feature names and SHAP values have mismatched lengths after all processing: "
                f"Expected {len(feature_names)}, got {len(shap_values)}"
            )

        shap_tuples = list(zip(feature_names, shap_values))
        shap_tuples.sort(key=lambda x: abs(x[1]), reverse=True)

        positive_features = []
        negative_features = []

        for feature, shap_value in shap_tuples:
            if shap_value > 0:
                positive_features.append({"feature_name": feature, "shap_value": shap_value})
            else:
                negative_features.append({"feature_name": feature, "shap_value": shap_value})

        return {
            "positive_features": positive_features[:5],
            "negative_features": negative_features[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")
