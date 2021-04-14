import numpy as np
import scipy.io as sio


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_grad_Jac_test_params(batch_size=1):
    data = sio.loadmat('..\\data\\PeaksData.mat')
    X_train = data["Yt"]
    Y_train = data["Ct"]
    x = data["Yt"][:, 0: batch_size]  # take a single x
    y = data["Ct"][:, 0: batch_size]
    n = x.shape[0]

    return x, y


if __name__ == '__main__':
    get_grad_Jac_test_params()
