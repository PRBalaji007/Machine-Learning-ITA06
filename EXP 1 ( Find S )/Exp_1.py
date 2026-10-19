
# EXPERIMENT 1: FIND-S ALGORITHM

import pandas as pd

# Training dataset
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High',   'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Cool', 'Change', 'Yes']
]

columns = [
    'Sky', 'AirTemp', 'Humidity', 'Wind',
    'Water', 'Forecast', 'EnjoySport'
]

df = pd.DataFrame(data, columns=columns)
print("Training Dataset:")
print(df.to_string(index=False))

# FIND-S algorithm
def find_s(dataset):
    attributes = dataset.iloc[:, :-1].values
    labels = dataset.iloc[:, -1].values

    # Start with the first positive example
    hypothesis = None

    for example, label in zip(attributes, labels):
        if label == 'Yes':
            if hypothesis is None:
                hypothesis = list(example)
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != example[i]:
                        hypothesis[i] = '?'

            print("Current hypothesis:", hypothesis)

    return hypothesis

final_hypothesis = find_s(df)

print("\nFinal Most Specific Hypothesis:")
print(final_hypothesis)
