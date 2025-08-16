# First, you need to import the necessary functions from scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# In a real scenario, you would get y_test from your data split
# and y_pred from your model's predictions.
# For this example, let's use some placeholder data to show how it works.
# Let's imagine our model's predictions (y_pred) and the actual values (y_test)
# The values are 0 (no stroke) and 1 (stroke)
# y_test represents the ground truth, y_pred represents the model's guess
y_test = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
y_pred = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0]

# Now, we'll calculate the metrics
print("--- Model Performance Metrics ---")

# 1. Accuracy Score
# This measures the overall percentage of correct predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 2. Precision Score
# This measures the proportion of positive identifications that were actually correct.
# In a medical context, it tells you how trustworthy your model's "stroke" predictions are.
# A high precision means when the model says "stroke", it's usually right.
# We need to specify `pos_label=1` because 'stroke' is our positive class.
precision = precision_score(y_test, y_pred, pos_label=1)
print(f"Precision: {precision:.2f}")

# 3. Recall Score
# This measures the proportion of actual positives that were identified correctly.
# In a medical context, it tells you how good your model is at finding all the
# patients who actually have a stroke. A high recall means it's less likely to miss a case.
recall = recall_score(y_test, y_pred, pos_label=1)
print(f"Recall: {recall:.2f}")

# 4. Confusion Matrix (for more detail)
# This gives you a full breakdown of the TP, TN, FP, and FN.
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
# The output matrix is structured as follows:
# [[TN, FP],
#  [FN, TP]]

# To make the matrix easier to read:
tn, fp, fn, tp = conf_matrix.ravel()
print(f"\nTrue Positives (TP): {tp}")   # Correctly predicted a stroke
print(f"True Negatives (TN): {tn}")   # Correctly predicted no stroke
print(f"False Positives (FP): {fp}") # Incorrectly predicted a stroke (Type I error)
print(f"False Negatives (FN): {fn}") # Incorrectly predicted no stroke (Type II error)
