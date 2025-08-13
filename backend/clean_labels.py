import pandas as pd

# Load your combined dataset
df = pd.read_csv("data/combined_disease_dataset.csv")

# Clean up "Yes"/"No" → 1/0 and fill missing values with 0
cleanup_map = {"Yes": 1, "No": 0}
disease_columns = ['has_diabetes', 'has_heart_disease', 'has_hypertension']

for col in disease_columns:
    df[col] = df[col].replace(cleanup_map)
    df[col] = df[col].fillna(0).astype(int)

# Save cleaned version
df.to_csv("data/cleaned_disease_dataset.csv", index=False)
print("✅ Cleaned and saved to cleaned_disease_dataset.csv")
