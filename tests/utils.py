import numpy as np
import scipy.io as sio


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_grad_Jac_test_params(batch_size=4):
    data = sio.loadmat('..\\data\\PeaksData.mat')
    X_train = data["Yt"]
    Y_train = data["Ct"]
    m = X_train.shape[1]
    perm_indices = np.random.permutation(m)
    chosen_indices = perm_indices[0:batch_size]
    X = data["Yt"][:, chosen_indices]  # take a single x
    Y = data["Ct"][:, chosen_indices]

    return X, Y


if __name__ == '__main__':
    get_grad_Jac_test_params()
