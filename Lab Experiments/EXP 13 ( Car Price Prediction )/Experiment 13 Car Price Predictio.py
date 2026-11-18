# Experiment 13: Car Price Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate sample car dataset
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "year": np.random.randint(2010, 2025, n),
    "kms_driven": np.random.randint(5000, 150001, n),
    "fuel": np.random.choice(["Petrol", "Diesel", "CNG"], n),
    "transmission": np.random.choice(["Manual", "Automatic"], n),
    "owner": np.random.randint(1, 4, n)
})

# Generate sample prices in Indian Rupees
age = 2025 - data["year"]
fuel_bonus = data["fuel"].map({
    "Petrol": 0, "Diesel": 80000, "CNG": -30000
})
transmission_bonus = data["transmission"].map({
    "Manual": 0, "Automatic": 100000
})

data["price"] = (
    900000 - age * 45000 - data["kms_driven"] * 2
    - (data["owner"] - 1) * 60000 + fuel_bonus
    + transmission_bonus + np.random.normal(0, 50000, n)
).clip(lower=50000)

X = data.drop("price", axis=1)
y = data["price"]

# Encode categorical features
preprocessor = ColumnTransformer(
    [("categorical", OneHotEncoder(handle_unknown="ignore"),
      ["fuel", "transmission"])],
    remainder="passthrough"
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Regression model
model = make_pipeline(
    preprocessor,
    RandomForestRegressor(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Car Price Prediction Results")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Car Price (₹)")
plt.ylabel("Predicted Car Price (₹)")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.show()
