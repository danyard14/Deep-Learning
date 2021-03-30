import numpy as np
from numpy import log, exp, divide
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio


def get_ettas(X, W, m, b):
    ettas = np.zeros(m)
    for j in range(m):
        x_j = X[:, j]
        candidates = x_j @ W + b
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


def soft_max_regression(X: np.array, Y: np.array, W: np.array, b: np.array):
    """
    description..
    :param X: a matrix of nxm where n is the dimension of x and m is the number of examples x
    :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
    :param W: a matrix nxl where n is the dimension of x and l is the number of labels
    :param b: a vector of size 1*l, where b[i] is the bias of w_i
    :return: int loss
    """
    # TODO:
    #  2. add bias

    n, m = X.shape
    l = W.shape[1]

    X_t = X.T
    loss = 0
    right_sum = 0
    ettas_vector = get_ettas(X, W, m, b)
    W_grads = np.zeros((n, l))  # each column j will be the grad with respect to w_j
    b_grad = np.zeros(l)
    # TODO: check if grads is not nxl (w has dimension nx1)

    for j in range(l):
        w_j = W[:, j]
        right_sum += exp(X_t @ w_j + b[j] - ettas_vector)

    for k in range(l):
        w_k = W[:, k]
        c_k = Y[k, :]
        diag_v_inv_u = divide(exp(X_t @ w_k + b[k] - ettas_vector), right_sum)
        loss += c_k.T @ log(diag_v_inv_u)
        W_grads[:, k] = (1 / m) * X @ (diag_v_inv_u - c_k)

    return - loss / m, W_grads


def get_acc(X, Y, W):
    XW = X.T @ W
    preds = np.argmax(XW, axis=1)
    true_labels = np.argmax(Y, axis=0)
    acc = sum(preds != true_labels)
    return acc / Y.shape[1]


def SGD(X, Y, W, lr=0.1, epchos=100, batch_size=32):
    m = X.shape[1]
    losses = np.zeros(epchos)
    accuracy = np.zeros(epchos)
    for i in range(epchos):
        if i % 25 == 0:
            lr /= 10
        perm_indices = np.random.permutation(m)
        loss = 0
        for j in range(0, m, batch_size):
            X_batch = X[:, perm_indices[j:j + batch_size]]
            Y_batch = Y[:, perm_indices[j:j + batch_size]]
            loss, grad = soft_max_regression(X_batch, Y_batch, W)
            W -= lr * grad
            bias
        losses[i] = loss
        accuracy[i] = get_acc(X, Y, W)
        ## validate

        print(f"epcoch = {i}, loss = {loss}, accuracy = {accuracy[i]} grad = {grad}")

    plt.plot(np.arange(0, epchos, 1), accuracy)
    plt.plot(np.arange(0, epchos, 1), losses)
    plt.show()
    return losses, W


def run_grad_test_soft_max():
    w = np.array([[1, -1], [0.5, 2]])
    return grad_test_soft_max(X_train, Y_train, w)


def test_SGD(X, Y):
    # W = np.ones((2, 2))
    W = np.random.random((2, 2))

    losses, W = SGD(X, Y, W)


if __name__ == '__main__':
    data = sio.loadmat('SwissRollData.mat')
    Y_train = data["Ct"]
    Y_validation = data["Cv"]
    X_train = data["Yt"]
    X_validation = data["Yv"]
    # run_grad_test_soft_max()
    test_SGD(X_train, Y_train)
