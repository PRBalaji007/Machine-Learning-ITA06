# Experiment 9: Comparison of Linear and Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset and select BMI as input feature
diabetes = load_diabetes()
X = diabetes.data[:, [2]]
y = diabetes.target

print("Diabetes Dataset Loaded Successfully")
print("Total Samples:", len(X))
print("Input Feature: BMI")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# Polynomial Regression (degree 2)
poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

# Evaluate both models
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)
poly_mse = mean_squared_error(y_test, poly_pred)
poly_r2 = r2_score(y_test, poly_pred)

print("\nLinear Regression Results")
print("Mean Squared Error:", round(linear_mse, 2))
print("R2 Score:", round(linear_r2, 4))

print("\nPolynomial Regression Results (Degree 2)")
print("Mean Squared Error:", round(poly_mse, 2))
print("R2 Score:", round(poly_r2, 4))

print("\nModel Comparison")
if poly_mse < linear_mse:
    print("Polynomial Regression has lower MSE.")
elif linear_mse < poly_mse:
    print("Linear Regression has lower MSE.")
else:
    print("Both models have equal MSE.")

# Plot actual data and regression curves
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.35, label="Actual Data")
plt.plot(X_plot, linear_model.predict(X_plot),
         label="Linear Regression")
plt.plot(X_plot, poly_model.predict(X_plot),
         label="Polynomial Regression (Degree 2)")
plt.xlabel("BMI (Standardized)")
plt.ylabel("Disease Progression")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.tight_layout()
plt.show()
