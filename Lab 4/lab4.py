import math
from collections import Counter
import pandas as pd


def calculate_entropy(data):

    entropy = 0.0
    freq = Counter(data)
    total_count = sum(freq.values())

    for count in freq.values():
        prob = count/total_count
        entropy -= prob*math.log2(prob)

    return entropy


df = pd.read_csv('Lab 2/titanic.csv')
columns = df.columns.tolist()
print(f'Columns in the dataset: {columns}')

col = input('Enter the column to calculate entropy for:')

if col in columns:
    data = df[col]
    entropy = calculate_entropy(data)
    print(f'Entropy for the {col} column is: {entropy}')
else:
    print('Column doesnt exist')
