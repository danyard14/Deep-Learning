from network import NeuralNetwork


class SGD:
    def __init__(self, net: NeuralNetwork, lr=0.001):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            layer.W = layer.W - self.lr * layer.dW
            layer.b = layer.b - self.lr * layer.db
