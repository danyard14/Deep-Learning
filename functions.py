import numpy as np
from numpy import log, exp, divide


def get_ettas(X, W, m):
    ettas = np.zeros(m)
    for j in range(m):
        x_j = X[:, j]
        candidates = x_j @ W
        ettas[j] = np.max(candidates)
    return ettas


def soft_max_regression(X: np.array, Y: np.array, W: np.array):
    """
    description..
    :param X: a matrix of nxm where n is the dimension of x and m is the number of examples x
    :param Y: a vector of size 1xm, where Y[i] is in [1,...,l]
    :param W: a matrix nxl where n is the dimension of x and l is the number of labels
    :return: int loss
    """
    # TODO: add etta to calculation
    m = X.shape[1]
    l = W.shape[1]

    X_t = X.T
    loss = 0
    right_sum = 0
    ettas_vector = get_ettas(X, W, m)
    grads = np.zeros((m, l))   # each column j will be the grad in respect to w_j

    for j in range(l):
        w_j = W[:, j]
        right_sum += exp(X_t @ w_j - ettas_vector)

    for k in range(l):
        w_k = W[:, k]
        c_k = np.array([1 if y == k else 0 for y in Y])
        diag_v_inv_u = divide(exp(X_t @ w_k - ettas_vector), right_sum)
        loss += c_k.T @ log(diag_v_inv_u)
        grads[:, k] = (1/m)*X @ (diag_v_inv_u - c_k)

    return - loss / m, grads


if __name__ == '__main__':
    pass
