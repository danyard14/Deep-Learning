import os
import matplotlib.pyplot as plt
from functions import soft_max_regression
from tests.utils import *
import scipy.io as sio
from layer import *


def grad_test_soft_max_all_weights(X: np.array, Y: np.array, W: np.array, b: np.array):
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
    n = X.shape[0]
    if len(X.shape) > 1:
        m = X.shape[1]
    else:
        m = 1
    d = normalize(np.random.rand(*W.shape))
    fw, grads, _ = soft_max_regression(X, Y, W, b)
    for i, epsilon in enumerate(epsilons):
        W_diff = W.copy()
        W_diff += d * epsilon
        fw_epsilon = soft_max_regression(X, Y, W_diff, b)[0]
        diff[i] = abs(fw_epsilon - fw)
        d_flat = d.reshape(-1, 1)
        grads_flat = grads.reshape(-1, 1)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * d_flat.T @ grads_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('weights Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def grad_test_soft_max_all_W_layer(X: np.array, Y: np.array):
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
    n = X.shape[0]
    l = Y.shape[0]
    if len(X.shape) > 1:
        m = X.shape[1]
    else:
        m = 1
    soft_max_layer = SoftMaxLayer(n, l)
    d = normalize(np.random.rand(*soft_max_layer.W.shape))
    W_orig = soft_max_layer.W.copy()
    out = soft_max_layer.forward(X)
    fw = soft_max_layer.soft_max(out, Y)
    grad_w = soft_max_layer.dW
    for i, epsilon in enumerate(epsilons):
        W_diff = W_orig.copy()
        W_diff += d * epsilon
        soft_max_layer.W = W_diff
        out_epsilon = soft_max_layer.forward(X)
        fw_epsilon = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fw_epsilon - fw)
        d_flat = d.reshape(-1, 1)
        grads_flat = grad_w.reshape(-1, 1)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * d_flat.T @ grads_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('weights Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def grad_test_soft_max_all_X_layer(X: np.array, Y: np.array):
    iter_num = 20
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    n = X.shape[0]
    l = Y.shape[0]
    soft_max_layer = SoftMaxLayer(n, l)
    d = normalize(np.random.rand(*X.shape))
    out = soft_max_layer.forward(X)
    fx = soft_max_layer.soft_max(out, Y)
    grad_x = soft_max_layer.dX
    for i, epsilon in enumerate(epsilons):
        X_diff = X.copy()
        X_diff += d * epsilon
        out_epsilon = soft_max_layer.forward(X_diff)
        fx_epsilon = soft_max_layer.soft_max(out_epsilon, Y)
        diff[i] = abs(fx_epsilon - fx)
        d_flat = d.reshape(-1, 1)
        grads_flat = grad_x.reshape(-1, 1)
        diff_grad[i] = abs(fx_epsilon - fx - epsilon * d_flat.T @ grads_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('X Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def grad_test_soft_max_bias(X: np.array, Y: np.array, W: np.array, b: np.array):
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
    l = len(b)
    d = normalize(np.random.rand(l))
    fw, _, grad_b = soft_max_regression(X, Y, W, b)
    for i, epsilon in enumerate(epsilons):
        b_diff = b.copy()
        b_diff += d * epsilon
        fw_epsilon = soft_max_regression(X, Y, W, b_diff)[0]
        diff[i] = abs(fw_epsilon - fw)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * d.T @ grad_b)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('bias Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


if __name__ == '__main__':
    # data = sio.loadmat('SwissRollData.mat')
    print(os.getcwd())
    data = sio.loadmat('..\\data\\PeaksData.mat')
    # data = sio.loadmat('GMMData.mat')
    Y_train = data["Ct"]
    Y_validation = data["Cv"]
    X_train = data["Yt"]
    X_validation = data["Yv"]
    # run_grad_test_soft_max()
    grad_test_soft_max_all_X_layer(*get_grad_Jac_test_params(batch_size=128))
