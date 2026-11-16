import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and prepare the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Y_train = np.eye(3)[y_train]

# Initialize the neural network: 4 input, 8 hidden, 3 output neurons
np.random.seed(42)
W1 = np.random.randn(4, 8) * np.sqrt(2 / 4)
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 3) * np.sqrt(2 / 8)
b2 = np.zeros((1, 3))

learning_rate = 0.05
epochs = 1500
m = len(X_train)

# Train the network using Backpropagation
for epoch in range(epochs):
    # Forward propagation
    Z1 = X_train @ W1 + b1
    A1 = np.maximum(0, Z1)             # ReLU
    Z2 = A1 @ W2 + b2

    exp_scores = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Backward propagation
    dZ2 = (probs - Y_train) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# Predict the test samples
Z1_test = X_test @ W1 + b1
A1_test = np.maximum(0, Z1_test)
Z2_test = A1_test @ W2 + b2
exp_test = np.exp(Z2_test - np.max(Z2_test, axis=1, keepdims=True))
prob_test = exp_test / np.sum(exp_test, axis=1, keepdims=True)
y_pred = np.argmax(prob_test, axis=1)

# Display evaluation results
print("Training completed successfully.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("ANN - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(3), iris.target_names, rotation=15)
plt.yticks(range(3), iris.target_names)
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.show()
