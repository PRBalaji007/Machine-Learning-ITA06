# Experiment 19: Bank Loan Approval Using Naive Bayes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
 
# Generate sample data for academic demonstration
rng = np.random.default_rng(19)
n = 700
data = pd.DataFrame({
    "age": rng.integers(21, 61, n),
    "annual_income": rng.integers(180000, 1800001, n),
    "credit_score": rng.integers(300, 851, n),
    "loan_amount": rng.integers(50000, 1500001, n)
})
 
decision_score = (
    data["annual_income"] / 10000
    + data["credit_score"] * 2
    + data["age"] * 5
    - data["loan_amount"] / 5000
    + rng.normal(0, 35, n)
)
data["loan_approved"] = (decision_score > 650).astype(int)
 
X = data[["age", "annual_income", "credit_score", "loan_amount"]]
y = data["loan_approved"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
 
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(
    y_test, y_pred, target_names=["Rejected", "Approved"]
))
print("Note: Generated sample data is for academic demonstration only.")
