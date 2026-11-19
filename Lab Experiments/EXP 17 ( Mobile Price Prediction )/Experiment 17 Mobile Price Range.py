# Experiment 17: Mobile Price Range Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
 
# Generate sample data for demonstration
rng = np.random.default_rng(12)
n = 600
data = pd.DataFrame({
    "battery_power": rng.integers(500, 2001, n),
    "ram": rng.integers(512, 8193, n),
    "px_height": rng.integers(400, 2001, n),
    "px_width": rng.integers(600, 2501, n),
    "int_memory": rng.integers(4, 257, n),
    "mobile_wt": rng.integers(80, 221, n)
})
 
score = (data["ram"] / 2048 + data["battery_power"] / 1000
         + data["px_height"] * data["px_width"] / 1500000
         + data["int_memory"] / 100 - data["mobile_wt"] / 250)
data["price_range"] = pd.qcut(
    score.rank(method="first"), 4, labels=[0, 1, 2, 3]
).astype(int)
 
X = data.drop("price_range", axis=1)
y = data["price_range"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
 
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions))
 
pd.Series(model.feature_importances_, index=X.columns).sort_values().plot(
    kind="barh", figsize=(8, 5), title="Mobile Feature Importance"
)
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
print("Note: Generated sample data is used, not a real market dataset.")
