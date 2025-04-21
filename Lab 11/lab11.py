import numpy as np
import skfuzzy as fuzz

temp_range = np.arange(0, 51, 1)

cold_mf = fuzz.trimf(temp_range, [0, 0, 20])
warm_mf = fuzz.trimf(temp_range, [15, 25, 35])
hot_mf = fuzz.trimf(temp_range, [30, 50, 50])


def predict(temp):

    labels = ['cold', 'warm', 'hot']
    membership_functions = [cold_mf, warm_mf, hot_mf]
    output = {}

    for label, mf in zip(labels, membership_functions):
        output[label] = round(
            float(fuzz.interp_membership(temp_range, mf, temp)), 2)

    return output


for _ in range(5):

    temp = round(np.random.uniform(0, 50), 2)
    print(f'Temperature: {temp} Fuzzy Membership: {predict(temp)}')
