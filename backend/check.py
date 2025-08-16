# Import the pandas library for data manipulation and analysis
import pandas as pd

# --- Step 1: Define the list of dataset filenames ---
# IMPORTANT: Replace these placeholder filenames with the actual names of your four dataset files.
# Make sure the files are in the same directory as this script.
dataset_files = [
    'data/kidneydisease.csv',
    'data/diabetes.csv',
    'data/heart.csv',
    'data/hypertension_dataset.csv'
]

# --- Step 2: Loop through each dataset and print information ---
for file_name in dataset_files:
    try:
        # Load the dataset from the CSV file
        df = pd.read_csv(file_name)

        # --- Section Header for Clarity ---
        print(f"============================================================")
        print(f"           ANALYZING DATASET: {file_name}")
        print(f"============================================================")

        # --- Get a general overview of the data ---
        print("\n--- First 5 rows of the dataset ---")
        print(df.head())

        # Get the number of rows and columns (shape of the DataFrame)
        print("\n--- Shape of the dataset (rows, columns) ---")
        print(df.shape)

        # --- Check for data types and missing values ---
        # The .info() method shows column names, non-null counts, and data types
        print("\n--- Detailed information (data types & non-null counts) ---")
        df.info()

        # A more direct way to count missing values
        print("\n--- Count of missing values in each column ---")
        print(df.isnull().sum())

        # --- Get descriptive statistics for numerical columns ---
        # .describe() provides statistical summary for numerical columns
        print("\n--- Descriptive statistics for numerical columns ---")
        print(df.describe())

        # Add a separator to make the output for each file distinct
        print("\n\n")

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        print("Please make sure the file name is correct and it's in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred while processing '{file_name}': {e}")

