import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_acc(out, Y):
    preds = np.argmax(out, axis=0)
    true_labels = np.argmax(Y, axis=0)
    acc = sum(preds == true_labels)
    return acc / Y.shape[1]