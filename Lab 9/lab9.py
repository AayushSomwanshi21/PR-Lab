import numpy as np


class Hopfield():

    def __init__(self, size):
        self.weights = np.zeros((size, size))
        self.size = size

    def train(self, patterns):

        for p in patterns:
            self.weights += np.outer(p, p)
        self.weights /= self.size
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, steps=1):

        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)

        return pattern


hopfield = Hopfield(size=4)

patterns = np.array([[1, 1, 1, 1]])
noisy_pattern = np.array([[1, -1, 1, 1]])

hopfield.train(patterns)
print(f'Recalled Pattern: {hopfield.predict(noisy_pattern[0])}')
