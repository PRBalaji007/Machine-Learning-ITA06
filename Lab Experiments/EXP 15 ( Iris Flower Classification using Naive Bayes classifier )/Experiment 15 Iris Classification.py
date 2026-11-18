# Experiment 15: Iris Classification Using Naive Bayes

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Iris Dataset Loaded Successfully")
print("Number of Samples:", len(X))
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict test samples
y_pred = model.predict(X_test)

# Evaluate model
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

# Plot classification results
plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="viridis", s=60)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Classification Using Naive Bayes")
plt.tight_layout()
plt.show()
