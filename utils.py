import numpy as np


def normalize(v):
    """
    Normalize the vector
    :param v: vector to normalize
    :return normalized vector v
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_acc(out, Y):
    """
    Calculate the NN accuracy
    :param out: the output of the NN after forwarded X in each layer at NN
    :param Y: a matrix of size lxm, where Y[i,:] is c_i (indicator vector for label i)
    :return the accuracy NN percentage
    """
    preds = np.argmax(out, axis=0)
    true_labels = np.argmax(Y, axis=0)
    acc = sum(preds == true_labels)
    return acc / Y.shape[1]
