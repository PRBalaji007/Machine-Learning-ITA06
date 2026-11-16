# Experiment 5: K-Nearest Neighbours (K-NN)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Iris Dataset Loaded Successfully")
print("Total Samples:", len(X))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nKNN Model Training Completed")
print("Number of Neighbours (K): 5")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
plt.imshow(cm, cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(range(3), iris.target_names)
plt.yticks(range(3), iris.target_names)

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()