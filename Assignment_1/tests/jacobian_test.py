import matplotlib.pyplot as plt
from Assignment_1 import functions
from Assignment_1.layer import *
from test_utils import *


def jacobian_test_layer_X(X, iter_num=20):
    """
    Jacobian test for X parameter
    :param X: a matrix of size nxm, output of forward layer
    :param iter_num: a number of iterations for the gradient test
    :return: void
    """
    layer = Layer(2, 3, activation=functions.tanh)
    n, m = X.shape
    out_dimensions = layer.b.shape[0]
    U = normalize(np.random.rand(out_dimensions, m))

    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = normalize(np.random.rand(*X.shape))

    fx = np.dot(layer.forward(X).T, U).item()
    layer.backward(U)
    JacTu_X = layer.dX

    for i, epsilon in enumerate(epsilons):
        X_diff = X.copy()
        X_diff += d * epsilon
        fx_epsilon = np.dot(layer.forward(X_diff).T, U).item()
        d_flat = d.reshape(-1, 1)
        JacTu_X_flat = JacTu_X.reshape(-1, 1)

        diff[i] = abs(fx_epsilon - fx)
        diff_grad[i] = abs(fx_epsilon - fx - epsilon * d_flat.T @ JacTu_X_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('X Jacobian Test Results')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|', "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀJᵀu|"))
    plt.show()


def jacobian_test_layer_W(X, iter_num=20):
    """
    Jacobian test for weights parameter
    :param X: a matrix of size nxm, output of forward layer
    :param iter_num: a number of iterations for the gradient test
    :return: void
    """
    layer = Layer(2, 3, activation=functions.tanh)
    n, m = X.shape
    out_dimensions = layer.b.shape[0]
    U = normalize(np.random.rand(out_dimensions, m))
    original_W = layer.W.copy()

    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = normalize(np.random.rand(*layer.W.shape))
    fw = np.dot(layer.forward(X).T, U).item()
    layer.backward(U)
    JacTu_W = layer.dW

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
    plt.title('weights Jacobian Test Results')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|', "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀJᵀu|"))
    plt.show()


def jacobian_test_layer_b(X, iter_num=20):
    """
    Jacobian test for bias parameter
    :param X: a matrix of size nxm, output of forward layer
    :param iter_num: a number of iterations for the gradient test
    :return: void
    """
    layer = Layer(2, 3, activation=functions.tanh)
    n, m = X.shape
    out_dimensions = layer.b.shape[0]
    U = normalize(np.random.rand(out_dimensions, m))
    original_b = layer.b.copy()

    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    d = normalize(np.random.rand(*layer.b.shape))
    fb = np.dot(layer.forward(X).T, U).item()
    layer.backward(U)
    JacTu_b = layer.db
    for i, epsilon in enumerate(epsilons):
        b_diff = original_b.copy()
        b_diff += d * epsilon
        layer.b = b_diff
        fb_epsilon = np.dot(layer.forward(X).T, U).item()
        diff[i] = abs(fb_epsilon - fb)
        diff_grad[i] = abs(fb_epsilon - fb - epsilon * d.T @ JacTu_b)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('bias Jacobian Test Results')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|', "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀJᵀu|"))
    plt.show()


if __name__ == '__main__':
    test_layer = Layer(2, 3)
    x = np.random.rand(2, 1)
    jacobian_test_layer_X(x)
    jacobian_test_layer_W(x)
    jacobian_test_layer_b(x)
