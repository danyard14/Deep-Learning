import matplotlib.pyplot as plt
from test_utils import *
from Assignment_1.network import *


def grad_test_soft_max_weights_nn(X: np.array, Y: np.array, iter_num=20):
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
    out = net.forward_pass(X)
    fw, _ = soft_max_layer.soft_max(out, Y)
    original_Ws = [layer.W.copy() for layer in net.layers]
    grads = np.zeros((0, 1))
    net.backward_pass()

    # build flat array of all layer's grads with respect to W
    for i, layer in enumerate(net.layers):
        grads = np.concatenate((grads, layer.dW.copy().reshape(-1, 1)))

    for i, epsilon in enumerate(epsilons):
        ds = np.zeros((0, 1))
        for j, layer in enumerate(net.layers):
            current_d = normalize(np.random.rand(*layer.W.shape))
            net.layers[j].W = original_Ws[j] + current_d * epsilon
            ds = np.concatenate((ds, current_d.reshape(-1, 1)))
        out_epsilon = net.forward_pass(X)
        soft_max_layer = net.soft_max_layer
        fw_epsilon, _ = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fw_epsilon - fw)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * ds.T @ grads)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('weights Grad Test Results - whole network')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|',
                "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀ∇f|"))
    plt.show()


def grad_test_soft_max_bias_nn(X: np.array, Y: np.array, iter_num=20):
    """
    Gradiant test for bias parameter
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
    out = net.forward_pass(X)
    fb, _ = soft_max_layer.soft_max(out, Y)
    original_bs = [layer.b.copy() for layer in net.layers]
    grads_b = np.zeros((0, 1))
    net.backward_pass()

    # build flat array of all layer's grads with respect to b
    for i, layer in enumerate(net.layers):
        grads_b = np.concatenate((grads_b, layer.db.copy().reshape(-1, 1)))

    for i, epsilon in enumerate(epsilons):
        ds = np.zeros((0, 1))
        for j, layer in enumerate(net.layers):
            current_d = normalize(np.random.rand(*layer.b.shape))
            net.layers[j].b = original_bs[j] + current_d * epsilon
            ds = np.concatenate((ds, current_d.reshape(-1, 1)))
        out_epsilon = net.forward_pass(X)
        soft_max_layer = net.soft_max_layer
        fb_epsilon, _ = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fb_epsilon - fb)
        diff_grad[i] = abs(fb_epsilon - fb - epsilon * ds.T @ grads_b)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('bias Grad Test Results - whole network')
    plt.legend((f'|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x)|',
                "|\N{LATIN SMALL LETTER F WITH HOOK}(x + εd) - \N{LATIN SMALL LETTER F WITH HOOK}(x) - εdᵀ∇f|"))
    plt.show()


if __name__ == '__main__':
    # grad_test_soft_max_weights_nn(*get_grad_Jac_test_params())
    grad_test_soft_max_bias_nn(*get_grad_Jac_test_params())
