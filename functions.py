import numpy as np
from numpy import log, exp, divide
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio


def get_ettas(X, W, m):
    ettas = np.zeros(m)
    for j in range(m):
        x_j = X[:, j]
        candidates = x_j @ W
        ettas[j] = np.max(candidates)
    return ettas


# def get_class_vec(Y_vecs):
#     l, m = Y_vecs.shape
#     Y = np.zeros(m)
#     for i in m:
#         np.argmax()

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def grad_test_soft_max(X: np.array, Y: np.array, W: np.array):
    """
       description..
       :param X: a matrix of nxm where n is the dimension of x and m is the number of examples x
       :param Y: a vector of size 1xm, where Y[i] is in [1,...,l]
       :param W: a vector nx1 where n is the dimension of x
       :return: int loss
       """
    iter_num = 20
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    n, m = X.shape
    d = normalize(np.random.rand(n))
    fw, grads = soft_max_regression(X, Y, W)
    grad = grads[:, 0]  # grad of w_0
    for i, epsilon in enumerate(epsilons):
        W_diff = W.copy()
        W_diff[:, 0] += d * epsilon
        fw_epsilon = soft_max_regression(X, Y, W_diff)[0]
        diff[i] = abs(fw_epsilon - fw)
        diff_grad[i] = abs(fw_epsilon - fw - epsilon * d.T @ grad)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()


def soft_max_regression(X: np.array, Y: np.array, W: np.array):
    """
    description..
    :param X: a matrix of nxm where n is the dimension of x and m is the number of examples x
    :param Y: a vector of size 1xm, where Y[i] is in [1,...,l]
    :param W: a matrix nxl where n is the dimension of x and l is the number of labels
    :return: int loss
    """
    # TODO:
    #  2. add bias

    n, m = X.shape
    l = W.shape[1]

    X_t = X.T
    loss = 0
    right_sum = 0
    ettas_vector = get_ettas(X, W, m)
    grads = np.zeros((n, l))  # each column j will be the grad with respect to w_j
    # TODO: check if grads is not nxl (w has dimension nx1)

    for j in range(l):
        w_j = W[:, j]
        right_sum += exp(X_t @ w_j - ettas_vector)

    for k in range(l):
        w_k = W[:, k]
        c_k = Y[k, :]
        diag_v_inv_u = divide(exp(X_t @ w_k - ettas_vector), right_sum)
        loss += c_k.T @ log(diag_v_inv_u)
        grads[:, k] = (1 / m) * X @ (diag_v_inv_u - c_k)

    return - loss / m, grads


if __name__ == '__main__':
    data = sio.loadmat('SwissRollData.mat')
    Y_train = data["Ct"]
    Y_validation = data["Cv"]
    X_train = data["Yt"]
    X_validation = data["Yv"]
    w = np.array([[1, -1], [0.5, 2]])
    grad_test_soft_max(X_train, Y_train, w)
    plt.show()
