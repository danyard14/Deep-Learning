import scipy.io as sio
import matplotlib.pyplot as plt
from functions import get_acc
from network import *
from optimizers import *


def train_network(data_path: str, num_layers= 8, batch_size: int = 32, lr: int = 0.001, epochs: int = 2):
    data = sio.loadmat(data_path)
    X_train = data["Yt"]
    Y_train = data["Ct"]
    input_size = X_train.shape[0]
    m = X_train.shape[1]
    num_of_classes = Y_train.shape[0]
    net = NeuralNetwork(input_size, num_layers, num_of_classes,policy='else')

    optimizer = SGD(net, lr)

    losses = np.zeros(epochs)
    accuracy = np.zeros(epochs)

    net.train_mode()
    for i in range(epochs):
        perm_indices = np.random.permutation(m)
        for j in range(0, m, batch_size):
            X_batch = X_train[:, perm_indices[j:j + batch_size]]
            Y_batch = Y_train[:, perm_indices[j:j + batch_size]]

            out = net.forward_pass(X_batch)
            loss = net.soft_max_layer.soft_max(out, Y_batch)
            net.backward_pass()
            optimizer.step()

            losses[i] += loss
            accuracy[i] += get_acc(out, Y_batch)

        losses[i] /= (m // batch_size)
        accuracy[i] /= (m // batch_size)

        print(f"epochs = {i}, loss = {losses[i]}, accuracy = {accuracy[i]}")

    plt.plot(np.arange(0, epochs, 1), accuracy)
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend("accuracy")
    plt.title(f"accuracy : batchsize = {batch_size} lr = {lr}")
    plt.show()

    plt.plot(np.arange(0, epochs, 1), losses)
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend("loss")
    plt.title(f"loss: batchsize = {batch_size} lr = {lr}")
    plt.show()


if __name__ == '__main__':
    train_network("data/PeaksData.mat", num_layers=7, batch_size=100, epochs=250)
