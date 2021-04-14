import numpy as np
from numpy import log, exp, divide
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import math


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


# def grad_test_soft_max(X: np.array, Y: np.array, W: np.array, b: np.array):
#     """
#        description..
#        :param X: a matrix of nxm where n is the dimension of x and m is the number of examples x
#        :param Y: a vector of size 1xm, where Y[i] is in [1,...,l]
#        :param W: a matrix nxl where n is the dimension of x
#        :param b: a vector lx1
#        :return: int loss
#        """
#     iter_num = 20
#     diff = np.zeros(iter_num)
#     diff_grad = np.zeros(iter_num)
#     epsilons = [0.5 ** i for i in range(iter_num)]
#     n, m = X.shape
#     d = normalize(np.random.rand(n))
#     fw, grads, _ = soft_max_regression(X, Y, W, b)
#     grad = grads[:, 0]  # grad of w_0
#     for i, epsilon in enumerate(epsilons):
#         W_diff = W.copy()
#         W_diff[:, 0] += d * epsilon
#         fw_epsilon = soft_max_regression(X, Y, W_diff, b)[0]
#         diff[i] = abs(fw_epsilon - fw)
#         diff_grad[i] = abs(fw_epsilon - fw - epsilon * d.T @ grad)
#
#     plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
#     plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
#     plt.xlabel('epsilons')
#     plt.ylabel('difference')
#     plt.title('weights Grad Test Results')
#     plt.legend(("diff without grad", "diff with grad"))
#     plt.show()


# def grad_test_soft_max_weights(X: np.array, Y: np.array, W: np.array, b: np.array):
#     """
#           description..
#           :param X: a matrix of nxm where n is the dimension of x and m is the number of examples x
#           :param Y: a vector of size 1xm, where Y[i] is in [1,...,l]
#           :param W: a matrix nxl where n is the dimension of x
#           :param b: a vector lx1
#           :return: int loss
#           """
#     iter_num = 20
#     diff = np.zeros(iter_num)
#     diff_grad = np.zeros(iter_num)
#     epsilons = [0.5 ** i for i in range(iter_num)]
#     n, m = X.shape
#     d = normalize(np.random.rand(W.shape))
#     fw, grads, _ = soft_max_regression(X, Y, W, b)
#     grad = grads[:, 0]  # grad of w_0
#     for i, epsilon in enumerate(epsilons):
#         W_diff = W.copy()
#         W_diff += d * epsilon
#         fw_epsilon = soft_max_regression(X, Y, W_diff, b)[0]
#         diff[i] = abs(fw_epsilon - fw)
#         diff_grad[i] = abs(fw_epsilon - fw - epsilon * d.T @ grad)
#
#     plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
#     plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
#     plt.xlabel('epsilons')
#     plt.ylabel('difference')
#     plt.title('weights Grad Test Results')
#     plt.legend(("diff without grad", "diff with grad"))
#     plt.show()


# def grad_test_soft_max_bias(X: np.array, Y: np.array, W: np.array, b: np.array):
#     """
#        description..
#        :param X: a matrix of nxm where n is the dimension of x and m is the number of examples x
#        :param Y: a vector of size 1xm, where Y[i] is in [1,...,l]
#        :param W: a matrix nxl where n is the dimension of x
#        :param b: a vector lx1
#        :return: int loss
#        """
#     iter_num = 20
#     diff = np.zeros(iter_num)
#     diff_grad = np.zeros(iter_num)
#     epsilons = [0.5 ** i for i in range(iter_num)]
#     n, m = X.shape
#     l = len(b)
#     d = normalize(np.random.rand(l))
#     fw, _, grad_b = soft_max_regression(X, Y, W, b)
#     for i, epsilon in enumerate(epsilons):
#         b_diff = b.copy()
#         b_diff += d * epsilon
#         fw_epsilon = soft_max_regression(X, Y, W, b_diff)[0]
#         diff[i] = abs(fw_epsilon - fw)
#         diff_grad[i] = abs(fw_epsilon - fw - epsilon * d.T @ grad_b)
#
#     plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
#     plt.xlabel('epsilons')
#     plt.ylabel('difference')
#     plt.title('bias Grad Test Results')
#     plt.legend(("diff without grad", "diff with grad"))
#     plt.show()


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
    n = X.shape[0]
    if len(X.shape) > 1:
        m = X.shape[1]
    else:
        m = 1

    l = W.shape[1]

    X_t = X.T
    loss = 0
    right_sum = 0
    ettas_vector = get_ettas(X, W, m, b)
    # ettas_vector = np.zeros(m)
    W_grads = np.zeros((n, l))  # each column j will be the grad with respect to w_j
    b_grad = np.zeros(l)

    for j in range(l):
        w_j = W[:, j]
        right_sum += exp(X_t @ w_j + b[j] - ettas_vector)

    for k in range(l):
        w_k = W[:, k]
        c_k = Y[k, :]
        diag_v_inv_u = divide(exp(X_t @ w_k + b[k] - ettas_vector), right_sum)
        loss += c_k.T @ log(diag_v_inv_u)
        W_grads[:, k] = (1 / m) * X @ (diag_v_inv_u - c_k)

        b_grad[k] = (1 / m) * np.sum((diag_v_inv_u - c_k))

    return - loss / m, W_grads, b_grad


# def get_acc(X, Y, W, b):
#     XW = X.T @ W + b
#     preds = np.argmax(XW, axis=1)
#     true_labels = np.argmax(Y, axis=0)
#     acc = sum(preds == true_labels)
#     return acc / Y.shape[1]

def get_acc(out, Y):
    preds = np.argmax(out, axis=0)
    true_labels = np.argmax(Y, axis=0)
    acc = sum(preds == true_labels)
    return acc / Y.shape[1]


def lr_step_decay(epoch, init_lr):
    reduce_ratio = 0.5
    decay_steps = 20

    """
    Scheduler to reduce learning rate by reduce_ratio every decay_step epochs.
    :param epoch: current epoch
    :return: updated learning rate
    """
    new_lr = init_lr * math.pow(reduce_ratio, math.floor((1 + epoch) / decay_steps))
    return new_lr


def update_lr(curr_epoch, init_lr):
    """
    Update learning rate based on the selected scheduling method.
    :param curr_epoch: current epoch number
    :param n_epochs: total number of epochs
    :return: updated learning rate
    """

    lr = lr_step_decay(curr_epoch, init_lr)
    print(f'Updated learning rate using step decay policy: {lr}')
    return lr


def SGD(X, Y, W, b, lr=0.001, epchos=100, batch_size=128):
    m = X.shape[1]
    losses = np.zeros(epchos)
    accuracy = np.zeros(epchos)
    for i in range(epchos):
        #     if i % 25 == 0:
        #         lr /= 10
        perm_indices = np.random.permutation(m)
        loss = 0
        for j in range(0, m, batch_size):
            X_batch = X[:, perm_indices[j:j + batch_size]]
            Y_batch = Y[:, perm_indices[j:j + batch_size]]
            loss, grad, b_grad = soft_max_regression(X_batch, Y_batch, W, b)
            W = W - lr * grad
            b = b - lr * b_grad
        losses[i] = loss
        accuracy[i] = get_acc(X, Y, W, b)

        ## validate

        print(f"epcoch = {i}, loss = {loss}, accuracy = {accuracy[i]} grad = {grad}")

    plt.plot(np.arange(0, epchos, 1), accuracy)
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend("accuracy")
    plt.title(f"accuracy : batchsize = {batch_size} lr = {lr}")
    plt.show()

    plt.plot(np.arange(0, epchos, 1), losses)
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend("loss")
    plt.title(f"loss: batchsize = {batch_size} lr = {lr}")
    plt.show()

    return losses, W


def SGD2(X, Y, W, b, lr=1e-3, lr2=1e-3, epchos=100, batch_size=32, momentum=0.9):
    init_lr = lr
    n = X.shape[0]
    num_of_classes = Y.shape[0]
    m = X.shape[1]
    losses = np.zeros(epchos)
    accuracy = np.zeros(epchos)
    prev_velocity_w = np.zeros(W.shape)
    prev_velocity_b = np.zeros(b.shape)

    for epoch in range(epchos):
        #     if i % 25 == 0:
        #         lr /= 10
        perm_indices = np.random.permutation(m)
        loss = 0
        for j in range(0, m, batch_size):
            X_batch = X[:, perm_indices[j:j + batch_size]]
            Y_batch = Y[:, perm_indices[j:j + batch_size]]
            loss, W_grad, b_grad = soft_max_regression(X_batch, Y_batch, W, b)
            new_velocity_w = (momentum * prev_velocity_w) - (lr * W_grad + lr * lr2 * W_grad)
            new_velocity_b = (momentum * prev_velocity_b) - (lr * b_grad)
            W = W + new_velocity_w
            b = b + new_velocity_b

            prev_velocity_w = new_velocity_w
            prev_velocity_b = new_velocity_b

        losses[epoch] = loss
        accuracy[epoch] = get_acc(X, Y, W, b)
        lr = update_lr(epoch, init_lr)
        ## validate

        print(f"epcoch = {epoch}, loss = {loss}, accuracy = {accuracy[epoch]} W_grad = {W_grad}")

    plt.plot(np.arange(0, epchos, 1), accuracy)
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend("accuracy")
    plt.title(f"accuracy : batchsize = {batch_size} lr = {lr}")
    plt.show()

    plt.plot(np.arange(0, epchos, 1), losses)
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend("loss")
    plt.title(f"loss: batchsize = {batch_size} lr = {lr}")
    plt.show()

    return losses, W


def run_grad_test_soft_max():
    w = np.array([[1, 1], [0.5, 2]])
    b = np.array([1.0, -1.0])
    grad_test_soft_max(X_train, Y_train, w, b)
    grad_test_soft_max_bias(X_train, Y_train, w, b)


def test_SGD(X, Y):
    # best hyperparameters seem to be lr = 0.001,batch_size=32
    # W = np.ones((2, 2))
    n = X.shape[0]
    num_of_classes = Y.shape[0]
    W = np.random.random((n, num_of_classes))
    b = np.random.random(num_of_classes)
    batch_sizes = [32]
    lrs = [0.001]

    for batch_size in batch_sizes:
        for lr in lrs:
            SGD2(X, Y, W, b)


class ReLU:
    @staticmethod
    def activate(x):
        return np.maximum(0, x)

    @staticmethod
    def deriviative(x):
        f = lambda x: 1 if x >= 0 else 0
        vfunc = np.vectorize(f)

        return vfunc(x)


class tanh:
    @staticmethod
    def activate(x):
        return np.tanh(x)  # TODO: check if should right whole expression

    @staticmethod
    def deriviative(x):
        f = lambda X: 1 - np.tanh(X) ** 2
        return f(x)


if __name__ == '__main__':
    # data = sio.loadmat('SwissRollData.mat')
    data = sio.loadmat('data/PeaksData.mat')
    # data = sio.loadmat('GMMData.mat')
    Y_train = data["Ct"]
    Y_validation = data["Cv"]
    X_train = data["Yt"]
    X_validation = data["Yv"]
    # run_grad_test_soft_max()
    test_SGD(X_train, Y_train)
