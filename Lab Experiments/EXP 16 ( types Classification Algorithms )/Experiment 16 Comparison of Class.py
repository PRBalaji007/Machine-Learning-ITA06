# Experiment 16: Comparison of Classification Algorithms
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
 
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2,
    random_state=42, stratify=iris.target
)
 
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "Naive Bayes": GaussianNB()
}
 
scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores[name] = accuracy_score(y_test, predictions) * 100
    print(f"{name}: {scores[name]:.2f}%")
 
print("\nBest model:", max(scores, key=scores.get))
 
plt.figure(figsize=(9, 5))
plt.bar(scores.keys(), scores.values())
plt.ylabel("Accuracy (%)")
plt.title("Classification Algorithm Comparison")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 105)
plt.tight_layout()
plt.show()
