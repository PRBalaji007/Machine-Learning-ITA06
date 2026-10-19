
# EXPERIMENT 3: DECISION TREE USING ID3 ALGORITHM

import pandas as pd
import math
from collections import Counter

# STEP 1: Load the uploaded CSV file
file_path = '/content/play_tennis - play_tennis.csv'
df = pd.read_csv(file_path)

# Clean column names and text values
df.columns = df.columns.str.strip()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip().str.lower()

print("TRAINING DATASET:")
print(df.to_string(index=False))
print("\nDataset shape:", df.shape)

# Target column in your CSV
target = 'Target'
attributes = [col for col in df.columns if col != target]


# STEP 2: Calculate entropy
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )


# STEP 3: Calculate information gain
def information_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0

    for value in data[attribute].unique():
        subset = data[data[attribute] == value]

        weighted_entropy += (
            len(subset) / len(data)
        ) * entropy(subset[target])

    return total_entropy - weighted_entropy


# STEP 4: Implement the ID3 algorithm
def id3(data, attributes, target):
    labels = data[target]

    # Stop if all examples belong to the same class
    if labels.nunique() == 1:
        return labels.iloc[0]

    # Stop if no attributes remain
    if not attributes:
        return labels.mode()[0]

    # Select attribute with maximum information gain
    gains = {
        attribute: information_gain(data, attribute, target)
        for attribute in attributes
    }

    best_attribute = max(gains, key=gains.get)

    tree = {best_attribute: {}}
    remaining_attributes = [
        a for a in attributes if a != best_attribute
    ]

    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]

        if subset.empty:
            tree[best_attribute][value] = labels.mode()[0]
        else:
            tree[best_attribute][value] = id3(
                subset, remaining_attributes, target
            )

    return tree


# STEP 5: Display information gain
print("\nINFORMATION GAIN:")
for attribute in attributes:
    gain = information_gain(df, attribute, target)
    print(f"{attribute}: {gain:.4f}")

# Build the decision tree
decision_tree = id3(df, attributes, target)


# STEP 6: Display the decision tree
def display_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + str(tree).upper())
        return

    attribute = next(iter(tree))

    for value, subtree in tree[attribute].items():
        print(f"{indent}{attribute} = {value}")
        display_tree(subtree, indent + "  ")

print("\nGENERATED ID3 DECISION TREE:")
display_tree(decision_tree)


# STEP 7: Classify a new sample
def predict(tree, sample, default):
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    value = sample.get(attribute)

    if value not in tree[attribute]:
        return default

    return predict(tree[attribute][value], sample, default)


new_sample = {
    'Outlook': 'sunny',
    'Temperature': 'cool',
    'Humidity': 'high',
    'Wind': 'strong'
}

# Normalize the sample values
new_sample = {
    key: str(value).strip().lower()
    for key, value in new_sample.items()
}

default_class = df[target].mode()[0]
prediction = predict(decision_tree, new_sample, default_class)

print("\nNEW SAMPLE:")
print(new_sample)

print("\nPREDICTED CLASS:", prediction.upper())

print("\nRESULT: ID3 decision tree constructed successfully.")
print("The new sample has been classified using the decision tree.")
