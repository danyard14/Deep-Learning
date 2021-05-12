import matplotlib.pyplot as plt
from test_utils import *
from Assignment_1.layer import *


def grad_test_soft_max_X(X: np.array, Y: np.array, iter_num=20):
    """
    Gradiant test for X parameter
    :param X: a matrix of size nxm, output of forward layer
    :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
    :param iter_num: a number of iterations for the gradient test
    :return: void
    """
    X_orig = X.copy()
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    n = X.shape[0]
    l = Y.shape[0]
    soft_max_layer = SoftMaxLayer(n, l)
    d = normalize(np.random.rand(*X.shape))
    out = soft_max_layer.forward(X)
    fx, _ = soft_max_layer.soft_max(out, Y)
    grad_x = soft_max_layer.dX.copy()
    for i, epsilon in enumerate(epsilons):
        X_diff = X_orig.copy()
        X_diff += d * epsilon
        out_epsilon = soft_max_layer.forward(X_diff)
        fx_epsilon, _ = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fx_epsilon - fx)
        d_flat = d.reshape(-1, 1)
        grads_flat = grad_x.reshape(-1, 1)
        diff_grad[i] = abs(fx_epsilon - fx - epsilon * d_flat.T @ grads_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('X Grad Test Results')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|', "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀ∇f|"))
    plt.show()


def grad_test_soft_max_weights(X: np.array, Y: np.array, iter_num=20):
    """
    Gradiant test for Weights parameter
    :param X: a matrix of size nxm, output of forward layer
    :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
    :param iter_num: a number of iterations for the gradient test
    :return: void
    """
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    n = X.shape[0]
    l = Y.shape[0]
    soft_max_layer = SoftMaxLayer(n, l)
    d = normalize(np.random.rand(*soft_max_layer.W.shape))
    W_orig = soft_max_layer.W.copy()
    out = soft_max_layer.forward(X)
    fw, _ = soft_max_layer.soft_max(out, Y)
    grad_w = soft_max_layer.dW
    for i, epsilon in enumerate(epsilons):
        W_diff = W_orig.copy()
        W_diff += d * epsilon
        soft_max_layer.W = W_diff
        out_epsilon = soft_max_layer.forward(X)
        fw_epsilon, _ = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fw_epsilon - fw)
        d_flat = d.reshape(-1, 1)
        grads_flat = grad_w.reshape(-1, 1)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * d_flat.T @ grads_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('Weights Grad Test Results')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|', "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀ∇f|"))
    plt.show()


def grad_test_soft_max_bias(X: np.array, Y: np.array, iter_num=20):
    """
    Gradiant test for bias parameter
    :param X: a matrix of size nxm, output of forward layer
    :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
    :param iter_num: a number of iterations for the gradient test
    :return: void
    """
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    n, m = X.shape
    l = Y.shape[0]
    soft_max_layer = SoftMaxLayer(n, l)
    b_original = soft_max_layer.b.copy()
    d = normalize(np.random.rand(*soft_max_layer.b.shape))
    out = soft_max_layer.forward(X)
    fb, _ = soft_max_layer.soft_max(out, Y)
    grad_b = soft_max_layer.db
    for i, epsilon in enumerate(epsilons):
        b_diff = b_original.copy()
        b_diff += d * epsilon
        soft_max_layer.b = b_diff
        out_epsilon = soft_max_layer.forward(X)
        fb_epsilon, _ = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fb_epsilon - fb)
        diff_grad[i] = abs(fb_epsilon - fb - epsilon * d.T @ grad_b)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('bias Grad Test Results')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|', "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀ∇f|"))
    plt.show()


if __name__ == '__main__':
    grad_test_soft_max_X(*get_grad_Jac_test_params())
    grad_test_soft_max_weights(*get_grad_Jac_test_params())
    grad_test_soft_max_bias(*get_grad_Jac_test_params())
