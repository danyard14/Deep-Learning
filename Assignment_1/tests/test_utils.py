import numpy as np
import scipy.io as sio
from Assignment_1.utils import normalize


def get_grad_Jac_test_params(batch_size=4):
    data = sio.loadmat('../data/PeaksData.mat')
    X_train = data["Yt"]
    Y_train = data["Ct"]
    m = X_train.shape[1]
    perm_indices = np.random.permutation(m)
    chosen_indices = perm_indices[0:batch_size]
    X = data["Yt"][:, chosen_indices]
    Y = data["Ct"][:, chosen_indices]

    X_normalize = normalize(np.random.rand(*X.shape))

    return X_normalize, Y


if __name__ == '__main__':
    get_grad_Jac_test_params()
