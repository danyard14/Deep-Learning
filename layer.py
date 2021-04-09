import numpy as np
from numpy import log, exp, divide
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import math


class Layer:
    def __init__(self, in_dimensions, out_dimensions):
        self.X = None
        self.W = np.random.rand(out_dimensions, in_dimensions)
        self.b = np.random.rand(out_dimensions, 1)
        self.activation = np.tanh  # TODO: check if should right whole expression
        self.activation_deriviative = lambda X: 1 - np.tanh(X) ** 2
        self.dX = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        out = self.activation(self.W @ X + self.b)

        return out

    def backward(self, V):
        temp = np.multiply(self.activation_deriviative(self.W @ self.X + self.b), V)
        self.dX = self.W.T @ temp
        self.dW = temp @ self.X.T
        self.db = temp

        return self.dX, self.dW, self.db

