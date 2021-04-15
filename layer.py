import numpy as np
from numpy import log, exp, divide
from functions import ReLU, tanh
from utils import normalize
import math

class Layer:
    def __init__(self, in_dimensions, out_dimensions, activation=ReLU):
        """

        :param in_dimensions: dimensions of input
        :param out_dimensions: dimensions of output
        :param activation: activation function used by the layer
        """
        self.X = None
        np.random.seed(0)
        self.W = np.random.uniform(-1, 1, size=(out_dimensions, in_dimensions))
        self.b = np.zeros((out_dimensions, 1))
        self.activation = activation.activate
        self.activation_derivative = activation.deriviative
        self.dX = None  # derivative with respect to input
        self.dW = None  # derivative with respect to Weight
        self.db = None  # derivative with respect to bias
        self.train = True

    def forward(self, X):
        self.X = X.copy()
        out = self.activation(self.W @ X + self.b)
        return out

    def backward(self, V):
        temp = self.activation_derivative(self.W @ self.X + self.b) * V
        self.dX = self.W.T @ temp
        self.dW = temp @ self.X.T
        self.db = np.sum(temp, axis=1).reshape(-1, 1)

    def train_mode(self):
        self.train = True

    def eval_mode(self):
        self.train = False


class SoftMaxLayer(Layer):
    def __init__(self, in_dimensions, num_of_classes):
        super(SoftMaxLayer, self).__init__(in_dimensions, num_of_classes)
        self.activation = lambda X: X

    def forward(self, X):
        self.X = X.copy()
        out = self.activation(self.W @ X + self.b)

        return out

    def backward(self, V=None):
        pass

    def soft_max(self, net_out, Y):
        """
        description..
        :param net_out: a matrix of size nxm, output of  forward layer
        :param Y: a matrix of size lxm,
         where Y[i,:] is c_i (indicator vector for label i)
        :return: loss score, and probabilities matrix for each class,x
        """
        W = self.W.T
        n = self.X.shape[0]
        if len(self.X.shape) > 1:
            m = self.X.shape[1]
        else:
            m = 1
        l = self.W.shape[0]
        self.dW = np.zeros((l, n))
        self.db = np.zeros((l, 1))

        ettas_vector = get_ettas(self.X, W, m, self.b)
        scores = exp(net_out - ettas_vector)
        right_sum = np.sum(scores, axis=0)
        probabilities = scores / right_sum
        loss = np.sum(Y * np.log(probabilities))

        # if during training, calculate gradients of loss and save
        if self.train:
            self.dW = (1 / m) * (self.X @ (probabilities - Y).T).T
            self.db = (1 / m) * np.sum((probabilities - Y), axis=1).reshape(-1, 1)
            self.dX = (1 / m) * (W @ (probabilities - Y))

        return -loss / m, probabilities

def get_ettas(X, W, m, b):
    ettas = np.zeros(m)
    for j in range(m):
        x_j = X[:, j]
        candidates = x_j @ W + b
        ettas[j] = np.max(candidates)
    return ettas