# Experiment 18: Iris Classification Using Perceptron
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
 
iris = load_iris()
X = iris.data[:, :2]
y = (iris.target != 0).astype(int)  # Setosa=0, Non-Setosa=1
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
model = make_pipeline(
    StandardScaler(),
    Perceptron(max_iter=1000, eta0=0.1, random_state=42)
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
 
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(
    y_test, y_pred, target_names=["Setosa", "Non-Setosa"]
))
 
plt.figure(figsize=(7, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="coolwarm", s=60)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Perceptron: Setosa vs Non-Setosa")
plt.tight_layout()
plt.show()
