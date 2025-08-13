# Import the pandas library for data manipulation
import pandas as pd
import numpy as np

# --- Kidney Disease Dataset Cleaning ---
def clean_kidney_data(file_path):
    """
    Cleans the 'kidneydisease.csv' dataset by:
    1. Replacing non-standard missing value indicators with NaN.
    2. Dropping the 'id' column if it exists.
    3. Imputing missing numerical values with the mean.
    4. Converting categorical columns to a numeric format using one-hot encoding.
    """
    print("Cleaning kidney_disease dataset...")
    # Load the dataset
    df = pd.read_csv(file_path)

    # Use a try-except block to handle cases where the 'id' column is missing
    try:
        df.drop('id', axis=1, inplace=True)
    except KeyError:
        print("Warning: 'id' column not found in kidneydisease.csv, skipping drop.")

    # Replace specific non-standard missing values with numpy NaN
    df.replace({'\t?': np.nan, '?': np.nan, '\t': np.nan, 'ckd\t': 'ckd', 'notckd\t': 'notckd'}, inplace=True)

    # Convert columns to their appropriate data types
    # Manually convert columns that should be numeric but are read as objects
    numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_cols:
        # Check if the column exists before trying to convert
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Impute missing numerical values with the mean of the column
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    # One-hot encode the categorical columns
    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    
    # Filter for columns that actually exist in the dataframe before one-hot encoding
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    df = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True)

    # Save the cleaned data
    df.to_csv('data/kidney_cleaned.csv', index=False)
    print("Kidney data cleaned and saved to 'data/kidney_cleaned.csv'")

# --- Diabetes Dataset Cleaning ---
def clean_diabetes_data(file_path):
    """
    Cleans the 'diabetes.csv' dataset by:
    1. Replacing biologically impossible '0' values in certain columns with NaN.
    2. Imputing these missing values with the mean of their respective columns.
    """
    print("Cleaning diabetes dataset...")
    # Load the dataset
    df = pd.read_csv(file_path)

    # Replace '0' with NaN in columns where 0 is not a valid value
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_replace:
        df[col] = df[col].replace(0, np.nan)

    # Impute missing values with the mean
    for col in cols_to_replace:
        df[col].fillna(df[col].mean(), inplace=True)

    # Save the cleaned data
    df.to_csv('data/diabetes_cleaned.csv', index=False)
    print("Diabetes data cleaned and saved to 'data/diabetes_cleaned.csv'")

# --- Heart Disease Dataset Cleaning ---
def clean_heart_data(file_path):
    """
    Cleans the 'heart.csv' dataset by:
    1. One-hot encoding categorical columns.
    2. Converting boolean-like columns ('Y'/'N') to binary (1/0).
    """
    print("Cleaning heart disease dataset...")
    # Load the dataset
    df = pd.read_csv(file_path)

    # One-hot encode the specified categorical columns
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)

    # Map 'M'/'F' to 1/0 for 'Sex'
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

    # Map 'Y'/'N' to 1/0 for 'ExerciseAngina'
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    
    # Map 'Y'/'N' to 1/0 for 'HeartDisease' (if needed)
    df['HeartDisease'] = df['HeartDisease'].replace({'Yes': 1, 'No': 0})

    # Save the cleaned data
    df.to_csv('data/heart_cleaned.csv', index=False)
    print("Heart disease data cleaned and saved to 'data/heart_cleaned.csv'")

# --- Hypertension Dataset Cleaning ---
def clean_hypertension_data(file_path):
    """
    Cleans the 'hypertension_dataset.csv' by:
    1. One-hot encoding all categorical columns.
    """
    print("Cleaning hypertension dataset...")
    # Load the dataset
    df = pd.read_csv(file_path)

    # One-hot encode the categorical columns
    categorical_cols = ['BP_History', 'Medication', 'Exercise_Level', 'Smoking_Status', 'Family_History', 'Has_Hypertension']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save the cleaned data
    df.to_csv('data/hypertension_cleaned.csv', index=False)
    print("Hypertension data cleaned and saved to 'data/hypertension_cleaned.csv'")


# Execute the cleaning functions with updated file paths
clean_kidney_data('data/kidneydisease.csv')
clean_diabetes_data('data/diabetes.csv')
clean_heart_data('data/heart.csv')
clean_hypertension_data('data/hypertension_dataset.csv')
