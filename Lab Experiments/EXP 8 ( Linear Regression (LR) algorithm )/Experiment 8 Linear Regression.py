# Experiment 8: Linear Regression

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print("Diabetes Dataset Loaded Successfully")
print("Total Samples:", len(X))
print("Number of Features:", X.shape[1])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Training Completed")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 4))

# Display actual and predicted values
print("\nActual vs Predicted Values:")
for i in range(5):
    print("Actual:", round(y_test[i], 2),
          "Predicted:", round(y_pred[i], 2))

# Plot actual vs predicted values
plt.figure(figsize=(6, 4))

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()
plt.show()