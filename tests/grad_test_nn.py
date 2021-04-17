import matplotlib.pyplot as plt
from test_utils import *
from network import *


def grad_test_soft_max_weights(X: np.array, Y: np.array, iter_num=20):
    """
    Gradiant test for Weights parameter
    :param X: a matrix of size nxm, output of forward layer
    :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
    :param iter_num: a number of iterations for the gradient test
    """
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    n = X.shape[0]
    l = Y.shape[0]
    net = NeuralNetwork(n, 4, l, policy="increase")
    soft_max_layer = net.soft_max_layer
    d = normalize(np.random.rand(*soft_max_layer.W.shape))
    W_orig = soft_max_layer.W.copy()
    out = net.forward_pass(X)
    fw, _ = soft_max_layer.soft_max(out, Y)
    grad_w = soft_max_layer.dW
    for i, epsilon in enumerate(epsilons):
        W_diff = W_orig.copy()
        W_diff += d * epsilon
        soft_max_layer.W = W_diff
        out_epsilon = net.forward_pass(X)
        fw_epsilon, _ = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fw_epsilon - fw)
        d_flat = d.reshape(-1, 1)
        grads_flat = grad_w.reshape(-1, 1)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * d_flat.T @ grads_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('weights Grad Test Results - whole network')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|',
                "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀ∇f|"))
    plt.show()


def grad_test_soft_max_bias(X: np.array, Y: np.array, iter_num=20):
    """
    Gradiant test for bias parameter
    :param X: a matrix of size nxm, output of forward layer
    :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
    :param iter_num: a number of iterations for the gradient test
    """
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    n, m = X.shape
    l = Y.shape[0]
    net = NeuralNetwork(n, 4, l, policy="increase")
    soft_max_layer = net.soft_max_layer
    b_original = soft_max_layer.b.copy()
    d = normalize(np.random.rand(*soft_max_layer.b.shape))
    out = net.forward_pass(X)
    fb, _ = soft_max_layer.soft_max(out, Y)
    grad_b = soft_max_layer.db
    for i, epsilon in enumerate(epsilons):
        b_diff = b_original.copy()
        b_diff += d * epsilon
        soft_max_layer.b = b_diff
        out_epsilon = net.forward_pass(X)
        fb_epsilon, _ = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fb_epsilon - fb)
        diff_grad[i] = abs(fb_epsilon - fb - epsilon * d.T @ grad_b)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('bias Grad Test Results - whole network')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|',
                "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀ∇f|"))
    plt.show()


if __name__ == '__main__':
    grad_test_soft_max_weights(*get_grad_Jac_test_params())
    grad_test_soft_max_bias(*get_grad_Jac_test_params())
