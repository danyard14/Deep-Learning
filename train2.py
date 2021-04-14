import numpy as np
from numpy import log, exp, divide
import matplotlib.pyplot as plt
import scipy.io as sio

from functions import get_acc
from layer import *


def SGD(X, Y, layer, lr=0.001, epchos=100, batch_size=32):
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
            out = layer.forward(X_batch)
            loss = layer.soft_max(out, Y_batch)
            grad, b_grad = layer.backward()
            layer.W = layer.W - lr * grad
            layer.b = layer.b - lr * b_grad
            losses[i] += loss
            accuracy[i] += get_acc(out, Y_batch)

        losses[i] /= (m // batch_size)
        accuracy[i] /= (m // batch_size)

        ## validate

        print(f"epcoch = {i}, loss = {losses[i]}, accuracy = {accuracy[i]} grad = {grad}")

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

    return losses


def prepare_SGD(X, Y):
    # best hyperparameters seem to be lr = 0.001,batch_size=32
    # W = np.ones((2, 2))
    n = X.shape[0]
    num_of_classes = Y.shape[0]
    soft_max_layer = SoftMaxLayer(n, num_of_classes)
    # W = np.random.random((n, num_of_classes))
    # b = np.random.random(num_of_classes)
    batch_sizes = [32]
    lrs = [0.001]

    for batch_size in batch_sizes:
        for lr in lrs:
            SGD(X, Y, soft_max_layer)


if __name__ == '__main__':
    # data = sio.loadmat('SwissRollData.mat')
    data = sio.loadmat('data/PeaksData.mat')
    # data = sio.loadmat('GMMData.mat')
    Y_train = data["Ct"]
    Y_validation = data["Cv"]
    X_train = data["Yt"]
    X_validation = data["Yv"]
    # run_grad_test_soft_max()
    prepare_SGD(X_train, Y_train)
