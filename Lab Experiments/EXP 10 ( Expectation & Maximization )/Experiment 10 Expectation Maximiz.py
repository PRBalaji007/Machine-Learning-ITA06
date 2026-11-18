# Experiment 10: Expectation Maximization Algorithm

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data from two Gaussian distributions
np.random.seed(42)
data1 = np.random.normal(3, 1, 300)
data2 = np.random.normal(8, 1.5, 300)
X = np.concatenate([data1, data2])
np.random.shuffle(X)

# Initialize parameters for two Gaussian components
K = 2
N = len(X)
means = np.percentile(X, [25, 75])
variances = np.array([np.var(X), np.var(X)])
weights = np.array([0.5, 0.5])

# Expectation-Maximization iterations
for iteration in range(100):

    # E-Step: Calculate responsibilities
    probabilities = np.zeros((N, K))

    for k in range(K):
        variance = max(variances[k], 1e-6)
        probabilities[:, k] = (
            weights[k] / np.sqrt(2 * np.pi * variance)
            * np.exp(-((X - means[k]) ** 2) / (2 * variance))
        )

    responsibilities = probabilities / (
        probabilities.sum(axis=1, keepdims=True) + 1e-10
    )

    # M-Step: Update parameters
    Nk = responsibilities.sum(axis=0)

    new_means = (responsibilities * X[:, None]).sum(axis=0) / Nk
    new_variances = (
        responsibilities * (X[:, None] - new_means) ** 2
    ).sum(axis=0) / Nk
    new_weights = Nk / N

    # Check convergence
    if np.allclose(means, new_means, atol=1e-5):
        means = new_means
        variances = new_variances
        weights = new_weights
        break

    means = new_means
    variances = new_variances
    weights = new_weights

# Assign each data point to its most likely cluster
labels = np.argmax(responsibilities, axis=1)

print("Expectation Maximization Results")
print("Iterations:", iteration + 1)

for k in range(K):
    print(f"\nCluster {k + 1}")
    print("Mean:", round(means[k], 3))
    print("Variance:", round(variances[k], 3))
    print("Mixing Probability:", round(weights[k], 3))

# Plot the clustered data
plt.figure(figsize=(8, 5))
plt.scatter(X, np.zeros(N), c=labels, cmap="viridis", s=20)
plt.xlabel("Data Values")
plt.yticks([])
plt.title("Expectation Maximization - Two Gaussian Clusters")
plt.show()