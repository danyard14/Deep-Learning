import numpy as np
import scipy.io as sio


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_grad_Jac_test_params():
    data = sio.loadmat('..\\data\\PeaksData.mat')
    X_train = data["Yt"]
    Y_train = data["Ct"]
    x = data["Yt"][:, 0:128].reshape(-1,1)  # take a single x
    y = data["Ct"][:, 0:128].reshape(-1,1)
    n = x.shape[0]
    num_of_classes = y.shape[0]
    W = np.random.random((n, num_of_classes))
    b = np.random.random(num_of_classes)

    return x, y


if __name__ == '__main__':
    get_grad_Jac_test_params()
