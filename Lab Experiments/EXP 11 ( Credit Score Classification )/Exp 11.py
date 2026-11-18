
# Experiment 11: Credit Score Classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate sample credit score dataset
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "annual_income": np.random.randint(20000, 150001, n),
    "debt": np.random.randint(1000, 80001, n),
    "credit_utilization": np.random.uniform(0.05, 0.95, n),
    "late_payments": np.random.randint(0, 8, n),
    "credit_history_years": np.random.randint(1, 26, n)
})

# Generate sample labels: 1 = Good, 0 = Poor
score = (
    0.00004 * data["annual_income"]
    - 0.00003 * data["debt"]
    - 2.5 * data["credit_utilization"]
    - 0.45 * data["late_payments"]
    + 0.12 * data["credit_history_years"]
)

data["credit_score"] = (score > score.median()).astype(int)

X = data.drop("credit_score", axis=1)
y = data["credit_score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Credit Score Classification Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Poor Credit", "Good Credit"]
))

# Feature importance graph
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("Credit Score Classification - Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
