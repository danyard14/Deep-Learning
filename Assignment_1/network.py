from Assignment_1.layer import *
from Assignment_1.functions import *


class NeuralNetwork:
    def __init__(self, input_size, num_of_layers, num_of_classes, policy="constant", activation=ReLU):
        """
        :param input_size: dimensions of input
        :param num_of_layers: number of layers to create
        :param num_of_classes: number of labels
        :param policy: NN layers policy
        :param activation: activation function
        """
        if policy == "constant" or num_of_layers == 1:
            self.layers = [Layer(input_size, input_size, activation) for i in range(num_of_layers - 1)] + [
                SoftMaxLayer(input_size, num_of_classes)]
        elif policy == "loss":
            self.layers = [SoftMaxLayer(input_size, num_of_classes)]
        else:
            # creating a list of layers, where we increase in dimensions each time until the middle layer
            # then we start decreasing again until the final loss layer with output size num_of_classes
            self.layers = [Layer(input_size, 6, activation)] + [Layer(2 * (i + 2), 2 * (i + 3), activation) for i in
                                                                range(1, (num_of_layers) // 2)] \
                          + [Layer(2 * (i + 3), 2 * (i + 2), activation) for i in
                             range((num_of_layers) // 2 - 1, 0, -1)] \
                          + [SoftMaxLayer(6, num_of_classes)]

        self.soft_max_layer = self.layers[-1]

    def forward_pass(self, X):
        """
        Calculate X prediction, by evaluate X on each layer at the NN
        :param X: a matrix of size nxm
        :return an output matrix, after evaluation X on all the network's layers
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward_pass(self):
        """
        Update derivative of X, W, b for each layer at NN
        """
        prev_dx = None
        for layer in self.layers[::-1]:
            layer.backward(prev_dx)
            prev_dx = layer.dX.copy()

    def train_mode(self):
        for layer in self.layers:
            layer.train_mode()

    def eval_mode(self):
        for layer in self.layers:
            layer.eval_mode()


if __name__ == '__main__':
    pass
