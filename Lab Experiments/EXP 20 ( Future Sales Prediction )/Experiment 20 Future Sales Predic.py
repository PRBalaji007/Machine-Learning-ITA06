# Experiment 20: Future Sales Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# Generate sample monthly sales data
rng = np.random.default_rng(20)
months = np.arange(1, 37)
promotion = rng.integers(0, 2, len(months))
sales = (12000 + 350 * months + 2500 * promotion
         + rng.normal(0, 900, len(months)))
data = pd.DataFrame({
    "month": months, "promotion": promotion, "sales": sales
})
print(data.head())
 
X = data[["month", "promotion"]]
y = data["sales"]
X_train, X_test = X.iloc[:30], X.iloc[30:]
y_train, y_test = y.iloc[:30], y.iloc[30:]
 
model = LinearRegression()
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
 
print("MAE:", round(mean_absolute_error(y_test, test_predictions), 2))
print("MSE:", round(mean_squared_error(y_test, test_predictions), 2))
print("R2 Score:", round(r2_score(y_test, test_predictions), 3))
 
future_months = np.arange(37, 43)
future_data = pd.DataFrame({
    "month": future_months,
    "promotion": np.zeros(6, dtype=int)
})
future_sales = model.predict(future_data)
 
print("Future Sales Forecast:")
for month, value in zip(future_months, future_sales):
    print(f"Month {month}: {value:.2f}")
 
plt.figure(figsize=(8, 5))
plt.plot(months, sales, marker="o", label="Historical Sales")
plt.plot(future_months, future_sales, marker="s",
         linestyle="--", label="Forecast Sales")
plt.axvline(36, color="gray", linestyle=":")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction")
plt.legend()
plt.tight_layout()
plt.show()
print("Note: Generated sample sales data is used for demonstration.")
