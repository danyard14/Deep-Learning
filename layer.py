import numpy as np
from numpy import log, exp, divide


def get_ettas(X, W, m, b):
    ettas = np.zeros(m)
    for j in range(m):
        x_j = X[:, j]
        candidates = x_j @ W + b
        ettas[j] = np.max(candidates)
    return ettas


class Layer:
    def __init__(self, in_dimensions, out_dimensions, activation=np.tanh):
        self.X = None
        self.W = np.random.rand(out_dimensions, in_dimensions)
        self.b = np.random.rand(out_dimensions, 1)
        self.activation = np.tanh  # TODO: check if should right whole expression
        self.activation_deriviative = lambda X: 1 - np.tanh(X) ** 2
        self.dX = None
        self.dW = None
        self.db = None
        self.train = True

    def forward(self, X):
        self.X = X
        out = self.activation(self.W @ X + self.b)

        return out

    def backward(self, V):
        temp = np.multiply(self.activation_deriviative(self.W @ self.X + self.b), V)
        self.dX = self.W.T @ temp
        self.dW = temp @ self.X.T
        self.db = temp

    def train(self):
        self.train = True

    def eval(self):
        self.train = False


class SoftMaxLayer(Layer):
    def __init__(self, in_dimensions, num_of_classes):
        super(SoftMaxLayer, self).__init__(in_dimensions, num_of_classes)
        self.activation = lambda X: X

    def backward(self, V=None):
        pass

    def soft_max(self, net_out, Y):
        """
        description..
        :param net_out: a matrix of size lxm, where the i column are the scores for x_i
        :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
        :return: int loss
        """
        W = self.W.T  # since in Lectures this layer's W is (nxl, opposite of nn_layers)

        n = self.X.shape[0]
        if len(self.X.shape) > 1:
            m = self.X.shape[1]
        else:
            m = 1

        l = self.W.shape[0]

        ettas_vector = get_ettas(self.X, W, m, self.b)
        self.dW = np.zeros((l, n))  # each column j will be the grad with respect to w_j
        self.db = np.zeros((l, 1))

        scores = exp(net_out - ettas_vector)
        right_sum = np.sum(scores, axis=0)

        div = divide(scores, right_sum)
        loss = np.sum(Y * np.log(div))

        if self.train:
            self.dW = (1/m) * (self.X @ (div - Y).T).T
            self.db = (1/m) * np.sum((div - Y), axis=1).reshape(-1, 1)
            self.dX = (1/m) * W @ (div - Y)

        return -loss / m

    @staticmethod
    def softmax(outputLayer):
        # finding the maximum
        outputLayer -= np.max(outputLayer)
        # calculate softmax
        result = (np.exp(outputLayer).T / np.sum(np.exp(outputLayer), axis=1)).T
        return result

    def softmaxRegression(self, x_L, y_mat):
        # active softmax
        theta_L = self.W
        b_L = self.b

        scores = np.dot(np.transpose(x_L), theta_L.T) + b_L.T
        probs = self.softmax(scores)
        m = x_L.shape[1]

        cost = (-1 / m) * (np.sum(y_mat.T * np.log(probs)))
        grad_theta = (-1 / m) * (x_L @ (y_mat.T - probs))
        grad_b = -(1 / m) * np.sum(y_mat.T - probs, axis=0).reshape(-1,1)

        return cost, grad_theta, grad_b, probs


