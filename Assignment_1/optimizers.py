from Assignment_1.network import NeuralNetwork


class SGD:
    def __init__(self, net: NeuralNetwork, lr=0.001):
        """
        :param net: the netword
        :param lr: learning rate
        """
        self.net = net
        self.lr = lr

    def step(self):
        """
        Update every layer in the network by set learning rate and the derivative for W and bias
        """
        for layer in self.net.layers:
            layer.W = layer.W - self.lr * layer.dW
            layer.b = layer.b - self.lr * layer.db
