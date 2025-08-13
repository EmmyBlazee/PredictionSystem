import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import joblib
import os
import warnings
import numpy as np

# Suppress a known warning from SHAP and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Define a function to train, evaluate, and explain models for a given dataset ---
def train_and_explain_models(file_path, target_column, model_title):
    """
    Loads a cleaned dataset, trains a Random Forest, Gradient Boosting, and a
    VotingClassifier, evaluates them, generates a SHAP summary plot, and
    saves the final ensemble model.

    Args:
        file_path (str): The path to the cleaned CSV file, including its folder.
        target_column (str): The name of the target column.
        model_title (str): A descriptive title for the model being trained.
    """
    print(f"\n--- Training and Evaluating Models for {model_title} ---")
    
    # Load the cleaned dataset from the specified path
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
        return

    # --- Data Preprocessing: The key to fixing the SHAP error ---
    print("Preprocessing data to handle categorical, boolean, and missing values...")
    
    # Identify and convert categorical columns to numbers using one-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    
    # *** NEW: Robustly convert any remaining non-numeric columns to numeric. ***
    # This specifically addresses the boolean columns causing the SHAP error.
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"  --> Converting object column '{col}' to numeric...")
            # Use pd.to_numeric with errors='coerce' to convert non-numeric
            # values to NaN, which will be handled next.
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif df[col].dtype == 'bool':
            print(f"  --> Converting boolean column '{col}' to int...")
            # Convert boolean True/False to 1/0
            df[col] = df[col].astype(int)
    
    # Fill any remaining missing values with the mean of the column.
    # This ensures all data is numeric and no NaNs exist.
    df.fillna(df.mean(), inplace=True)
    
    # Separate features (X) and target (y)
    if target_column not in df.columns:
        print(f"Error: The target column '{target_column}' was not found after preprocessing.")
        return
        
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    # Final check to confirm all features are numeric before training
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        print("Error: After robust preprocessing, some features are still not numeric.")
        non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        print(f"Non-numeric columns remaining: {non_numeric_cols}")
        return
    else:
        print("All features successfully converted to numeric type.")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the individual models
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Combine the models using a VotingClassifier
    print("Training Voting Classifier (Ensemble Model)...")
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model)],
        voting='soft',
        weights=[0.5, 0.5]
    )
    ensemble_model.fit(X_train, y_train)
    
    # --- Model Evaluation ---
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    ensemble_pred = ensemble_model.predict(X_test)
    
    # Calculate and print accuracy
    print(f"\nAccuracy Scores for {model_title}:")
    print(f"  Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(f"  Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
    print(f"  Ensemble Model Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
    
    # --- Save the trained ensemble model ---
    model_folder = 'models'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        print(f"Created directory: '{model_folder}'")

    model_filename = f'{model_title.lower().replace(" ", "_")}_ensemble_model.joblib'
    model_path = os.path.join(model_folder, model_filename)
    joblib.dump(ensemble_model, model_path)
    print(f"\nEnsemble model saved to '{model_path}'")
    
    # --- SHAP for Model Explanation ---
    print("\nGenerating SHAP explanation for the Random Forest model within the Ensemble...")
    
    # Create a SHAP explainer for the trained RANDOM FOREST model (a component of the ensemble)
    explainer = shap.Explainer(rf_model, X_train)
    
    # Calculate SHAP values for the test data and disable the additivity check
    # to fix the TypeError caused by minor floating-point differences.
    shap_values = explainer(X_test, check_additivity=False)
    
    # Generate the SHAP summary plot
    print(f"Displaying SHAP summary plot for {model_title}. This may take a moment...")
    shap.summary_plot(shap_values, X_test, show=False)
    
    # Save the plot to a file
    plt.title(f'SHAP Summary Plot for {model_title} (Random Forest Component)')
    plot_filename = f'{model_title.lower().replace(" ", "_")}_shap_summary.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()
    print(f"SHAP summary plot saved as '{plot_filename}'")
    

# --- Execute the function for each dataset ---
# The file_path includes the 'data/' folder as you specified
train_and_explain_models(
    file_path='data/kidney_cleaned.csv',
    target_column='Chronic Kidney Disease: yes',
    model_title='Kidney Disease Prediction'
)

train_and_explain_models(
    file_path='data/diabetes_cleaned.csv',
    target_column='Outcome',
    model_title='Diabetes Prediction'
)

train_and_explain_models(
    file_path='data/heart_cleaned.csv',
    target_column='HeartDisease',
    model_title='Heart Disease Prediction'
)

train_and_explain_models(
    file_path='data/hypertension_cleaned.csv',
    target_column='Has_Hypertension_Yes',
    model_title='Hypertension Prediction'
)
