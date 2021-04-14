from layer import *


class NeuralNetwork:
    def __init__(self, input_size, num_of_layers, num_of_classes, policy="constant"):

        if policy == "constant" or num_of_layers == 1:
            self.layers = [Layer(input_size, input_size) for i in range(num_of_layers - 1)] + [
                SoftMaxLayer(input_size, num_of_classes)]

        else:
            self.layers = [Layer(input_size, 6)] + [Layer(2 * (i + 2), 2 * (i + 3)) for i in
                                                    range(1, num_of_layers // 2)] \
                          + [Layer(2 * (i + 3), 2 * (i + 2)) for i in range(num_of_layers // 2 - 1, 0, -1)] \
                          + [SoftMaxLayer(6, num_of_classes)]

        self.soft_max_layer = self.layers[-1]

    def forward_pass(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward_pass(self):
        prev_dx = None
        for layer in self.layers[::-1]:
            layer.backward(prev_dx)
            prev_dx = layer.dX

    def train_mode(self):
        for layer in self.layers:
            layer.train_mode()

    def eval_mode(self):
        for layer in self.layers:
            layer.eval_mode()

if __name__ == '__main__':
     pass