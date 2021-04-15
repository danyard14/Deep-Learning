import scipy.io as sio
import matplotlib.pyplot as plt
from network import *
from optimizers import *
from utils import get_acc


def train_network(data_path: str, num_layers=1, batch_size: int = 32, lr: int = 0.001, epochs: int = 3,
                  policy="increase"):
    data = sio.loadmat(data_path)
    X_train = data["Yt"]
    Y_train = data["Ct"]
    X_test = data["Yv"]
    Y_test = data["Cv"]
    input_size = X_train.shape[0]
    m = X_train.shape[1]
    num_of_classes = Y_train.shape[0]
    net = NeuralNetwork(input_size, num_layers, num_of_classes, policy=policy)
    optimizer = SGD(net, lr)
    losses = np.zeros(epochs)
    validation_accuracy = np.zeros(epochs)
    training_accuracy = np.zeros(epochs)
    best_loss = math.inf
    best_valid_acc = 0
    best_train_acc = 0
    for epoch in range(epochs):
        net.train_mode()
        perm_indices = np.random.permutation(m)
        for j in range(0, m, batch_size):
            X_batch = X_train[:, perm_indices[j:j + batch_size]]
            Y_batch = Y_train[:, perm_indices[j:j + batch_size]]

            out = net.forward_pass(X_batch)
            loss, probabilities = net.soft_max_layer.soft_max(out, Y_batch)
            net.backward_pass()
            optimizer.step()
            losses[epoch] += loss
            training_accuracy[epoch] += get_acc(probabilities, Y_batch)

        losses[epoch] /= (m // batch_size)
        training_accuracy[epoch] /= (m // batch_size)
        validation_accuracy[epoch] = validate(net, X_test, Y_test)
        best_loss = min(best_loss, losses[epoch])
        best_valid_acc = max(best_valid_acc, validation_accuracy[epoch])
        best_train_acc = max(best_train_acc, training_accuracy[epoch])

        print(f"epochs = {epoch}, loss = {losses[epoch]}, validation_accuracy = {validation_accuracy[epoch]}"
              f" train_accuracy = {training_accuracy[epoch]}")

    axis1.plot(np.arange(0, epochs, 1), training_accuracy)
    axis1.plot(np.arange(0, epochs, 1), validation_accuracy)
    axis1.set_xlabel("epochs")
    axis1.set_ylabel("score")
    axis1.legend(("training accuracy", "validation accuracy"))
    axis1.set_title(f"accuracy : batchsize = {batch_size} lr = {lr}")

    axis2.plot(np.arange(0, epochs, 1), losses)
    axis2.set_xlabel("epochs")
    axis2.set_ylabel("score")
    axis2.legend("loss")
    axis2.set_title(f"loss: batchsize = {batch_size} lr = {lr}")
    print(f"for batch size = {batch_size} and lr = {lr} we got loss = {best_loss},"
          f" valid_acc = {best_valid_acc} and train acc = {best_train_acc}")


def validate(net: NeuralNetwork, X_test, Y_test):
    net.eval_mode()
    out = net.forward_pass(X_test)
    _, probabilities = net.soft_max_layer.soft_max(out, Y_test)
    acc = get_acc(probabilities, Y_test)
    return acc


if __name__ == '__main__':
    Peaks = "data/PeaksData.mat"
    Swiss = "data/SwissRollData.mat"
    GMM = "data/GMMData.mat"
    # lrs = [0.001, 0.05]
    # batch_sizes = [32, 64]
    policies = ["increase"]
    num_of_layers = [4, 8, 12, 16]
    _, axis1 = plt.subplots(1, 1)
    _, axis2 = plt.subplots(1, 1)
    for policy in policies:
        for l in num_of_layers:
            train_network(Peaks, num_layers=l, policy=policy)
    plt.show()
