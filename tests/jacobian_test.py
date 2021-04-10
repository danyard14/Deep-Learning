import numpy as np
import matplotlib.pyplot as plt
from functions import soft_max_regression
from layer import *
from utils import *


def grad_test_soft_max(X: np.array, Y: np.array, W: np.array, b: np.array):
    """
       description..
       :param X: a matrix of nxm where n is the dimension of x and m is the number of examples x
       :param Y: a vector of size 1xm, where Y[i] is in [1,...,l]
       :param W: a matrix nxl where n is the dimension of x
       :param b: a vector lx1
       :return: int loss
       """
    iter_num = 20
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    n, m = X.shape
    d = normalize(np.random.rand(n))
    fw, grads, _ = soft_max_regression(X, Y, W, b)
    grad = grads[:, 0]  # grad of w_0
    for i, epsilon in enumerate(epsilons):
        W_diff = W.copy()
        W_diff[:, 0] += d * epsilon
        fw_epsilon = soft_max_regression(X, Y, W_diff, b)[0]
        diff[i] = abs(fw_epsilon - fw)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * d.T @ grad)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('weights Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def jacobian_test_layer_X(X, layer: Layer):
    n, m = X.shape
    out_dimensions = layer.b.shape[0]
    U = normalize(np.random.rand(out_dimensions, m))

    ###################
    iter_num = 20
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = normalize(np.random.rand(n))
    fx = np.dot(layer.forward(X).T, U).item()
    JacTu_X, _, _ = layer.backward(U)

    for i, epsilon in enumerate(epsilons):
        X_diff = X.copy()
        X_diff[:, 0] += d * epsilon
        fx_epsilon = np.dot(layer.forward(X_diff).T, U).item()
        diff[i] = abs(fx_epsilon - fx)
        diff_grad[i] = abs(fx_epsilon - fx - epsilon * d.T @ JacTu_X)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('X Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def jacobian_test_layer_W(X, layer: Layer):
    n, m = X.shape
    out_dimensions = layer.b.shape[0]
    U = normalize(np.random.rand(out_dimensions, m))
    original_W = layer.W.copy()

    ###################
    iter_num = 20
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = normalize(np.random.rand(*layer.W.shape))
    fw = np.dot(layer.forward(X).T, U).item()
    _, JacTu_W, _ = layer.backward(U)

    for i, epsilon in enumerate(epsilons):
        W_diff = original_W.copy()
        W_diff += d * epsilon
        layer.W = W_diff
        fw_epsilon = np.dot(layer.forward(X).T, U).item()
        diff[i] = abs(fw_epsilon - fw)
        d_flat = d.reshape(-1, 1)
        JacTu_W_flat = JacTu_W.reshape(-1, 1)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * d_flat.T @ JacTu_W_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('weights Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def jacobian_test_layer_b(X, layer: Layer):
    n, m = X.shape
    out_dimensions = layer.b.shape[0]
    U = normalize(np.random.rand(out_dimensions, m))
    original_b = layer.b.copy()

    ###################
    iter_num = 20
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = normalize(np.random.rand(out_dimensions))
    fb = np.dot(layer.forward(X).T, U).item()
    _, _, JacTu_b = layer.backward(U)

    for i, epsilon in enumerate(epsilons):
        b_diff = original_b.copy()
        b_diff[:, 0] += d * epsilon
        layer.b = b_diff
        fb_epsilon = np.dot(layer.forward(X).T, U).item()
        diff[i] = abs(fb_epsilon - fb)
        diff_grad[i] = abs(fb_epsilon - fb - epsilon * d.T @ JacTu_b)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('bias Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


if __name__ == '__main__':
    test_layer = Layer(2, 3)
    x = np.random.rand(2, 1)
    # jacobian_test_layer_X(x, test_layer)
    # jacobian_test_layer_W(x, test_layer)
    jacobian_test_layer_b(x, test_layer)
