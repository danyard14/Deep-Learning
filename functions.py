import numpy as np


class ReLU:
    @staticmethod
    def activate(x):
        return np.maximum(0, x)

    @staticmethod
    def deriviative(x):
        f = lambda t: 1 if t >= 0 else 0
        vfunc = np.vectorize(f)

        return vfunc(x)


class tanh:
    @staticmethod
    def activate(x):
        return np.tanh(x)

    @staticmethod
    def deriviative(x):
        f = lambda X: 1 - np.tanh(X) ** 2
        return f(x)
