import numpy as np


def step_function(x):
    return 1 if x >= 0 else 0


def train(x, y, learning_rate=0.1, epochs=10):

    weights = np.zeros(x.shape[1])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(x)):
            weighted_sum = np.dot(x[i], weights) + bias
            output = step_function(weighted_sum)

            error = y[i] - output

            weights += learning_rate*error*x[i]
            bias += learning_rate*error

    return weights, bias


def predict(x, weights, bias):
    output = np.dot(x, weights) + bias
    return step_function(output)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

weights, bias = train(x, y)

for sample in x:
    print(f'Input:{sample} Output:{predict(sample, weights, bias)}')
