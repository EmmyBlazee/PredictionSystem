# This script sends sample data to each of the disease prediction endpoints
# to verify that the backend API is working correctly.

import requests
import json

# Define the base URL of your FastAPI server
BASE_URL = "http://127.0.0.1:8000"

# Sample data payloads for each disease
# These payloads are complete and match the feature requirements of each model.

sample_data = {
    "heart": {
        "Age": 40,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up"
    },
    "diabetes": {
        "Pregnancies": 6,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 155.5,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    },
    "cardio": {
        "age": 50,
        "gender": 2,
        "height": 168,
        "weight": 62.0,
        "ap_hi": 110,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    },
    "kidney": {
        "age": 48.0,
        "bp": 80.0,
        "sg": 1.02,
        "al": 1.0,
        "su": 0.0,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "bgr": 121.0,
        "bu": 36.0,
        "sc": 1.2,
        "sod": 137.5,
        "pot": 4.6,
        "hemo": 15.4,
        "pcv": 44.0,
        "wc": 7800.0,
        "rc": 5.2,
        "htn": "yes",
        "dm": "yes",
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no"
    },
    "hypertension": {
        "age": 50,
        "gender": 2,
        "height": 168,
        "weight": 62.0,
        "ap_hi": 110,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
}

# Iterate through each disease and test its prediction endpoint
for disease, data in sample_data.items():
    url = f"{BASE_URL}/predict/{disease}"
    headers = {"Content-Type": "application/json"}
    
    print(f"\n--- Testing Endpoint: POST {url} ---")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status() # Raise an HTTPError for bad status codes (4xx or 5xx)
        
        # If the request was successful, print the status and response
        print(f"✅ Success! Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.HTTPError as err:
        # If there's an HTTP error, print the error details
        print(f"❌ HTTP Error for {disease}: {err}")
        print(f"Response Content: {response.text}")
    except requests.exceptions.RequestException as err:
        # Handle other request-related errors (e.g., connection errors)
        print(f"❌ Request Error for {disease}: {err}")
        print("Is the FastAPI server running? Please check 'uvicorn main:app --reload'")
