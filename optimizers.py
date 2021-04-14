import numpy as np
from numpy import log, exp, divide
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import math

# images, labels = batch
#
#             labels = labels.to(device='cuda')
#             images = images.to(device='cuda')
#
#             preds = detector(images)  # Pass Batch
#             loss = functional.cross_entropy(preds, labels)  # Calculate Loss
#
#             optimizer.zero_grad()
#             loss.backward()  # Calculate Gradients
#             optimizer.step()  # Update Weights
from layer import Layer


class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, layer: Layer):
        layer.W = layer.W - self.lr * layer.dW
        layer.b = layer.b - self.lr * layer.db



