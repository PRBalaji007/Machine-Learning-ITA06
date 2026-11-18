# Experiment 14: House Price Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate sample house dataset
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "area_sqft": np.random.randint(500, 4001, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 5, n),
    "age_years": np.random.randint(0, 31, n),
    "location": np.random.choice(["City", "Suburb", "Town"], n)
})

# Generate sample house prices in Indian Rupees
location_bonus = data["location"].map({
    "City": 1500000,
    "Suburb": 800000,
    "Town": 300000
})

data["price"] = (
    1000000 + data["area_sqft"] * 3500
    + data["bedrooms"] * 200000
    + data["bathrooms"] * 150000
    - data["age_years"] * 30000
    + location_bonus
    + np.random.normal(0, 300000, n)
).clip(lower=500000)

X = data.drop("price", axis=1)
y = data["price"]

# Encode categorical location feature
preprocessor = ColumnTransformer(
    [("location", OneHotEncoder(handle_unknown="ignore"), ["location"])],
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

print("House Price Prediction Results")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual House Price (₹)")
plt.ylabel("Predicted House Price (₹)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()
