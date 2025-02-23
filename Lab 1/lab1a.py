import numpy as np


class Neuron:

    def __init__(self, x):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def train(self, x, y, learning_rate=0.1, epochs=10):

        for epoch in range(epochs):
            for i in range(len(x)):
                weighted_sum = np.dot(x[i], self.weights) + self.bias
                output = self.step_function(weighted_sum)
                error = y[i] - output

                self.weights += learning_rate*error*x[i]
                self.bias += learning_rate*error

    def predict(self, x):

        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.step_function(weighted_sum)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

neuron = Neuron(x)
neuron.train(x, y)

for sample in x:
    print(f'Input: {sample} Output: {neuron.predict(sample)}')
