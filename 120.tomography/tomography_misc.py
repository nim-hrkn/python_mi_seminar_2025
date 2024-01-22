# scikit learnのコードです．文献[3]

# 以下ではoriginalのコードの名前をそのまま用いているので対応がわかりにくいですが
# - proj_operator = X
# - data = w
# - proj = y 
# という対応関係です

import numpy as np
from scipy import sparse

def _weights(x, dx=1, orig=0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y


def build_projection_operator(l_x, n_dir):
    """ Compute the tomography design matrix.

    Parameters
    ----------

    l_x : int
        linear size of image array

    n_dir : int
        number of angles at which projections are acquired.

    Returns
    -------
    p : sparse matrix of shape (n_dir l_x, l_x**2)
    """
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x ** 2)
    data_unravel_indices = np.hstack((data_unravel_indices,
                                      data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y

        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < l_x)

        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator

import matplotlib.pyplot as plt

def plot_images(w, w2, w_title, w2_title, titlefontsize=20, tickfontsize=15, filenames=None):
    """再構成した図をならべて表示する．

    Args:
        w (np.array): image 1
        w2 (np.array): image 2
        w_title (str): title 1
        w2_title (str): title 2
        titlefontsize (int, optional): title font size. Defaults to 20.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        filenames (Tuple[str]): filenames. Defaults to None.
    """
    plt.figure(figsize=(8, 3.3))
    plt.subplot(121)
    plt.imshow(w, cmap=plt.cm.gray, interpolation='nearest')
    plt.title(w_title, fontsize=titlefontsize)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(w2, cmap=plt.cm.gray, interpolation='nearest')
    plt.title(w2_title, fontsize=titlefontsize)
    plt.axis('off')
    plt.tight_layout()
    if filenames is not None:
        plt.savefig(filenames[0])
    plt.show()

    plt.figure(figsize=(8, 3.3))
    plt.subplot(121)
    plt.hist(w.reshape(-1), bins=100,)
    plt.title(w_title, fontsize=titlefontsize)
    plt.tick_params(axis = 'x', labelsize =tickfontsize)
    plt.tick_params(axis = 'y', labelsize =tickfontsize)   
    plt.subplot(122)
    plt.hist(w2.reshape(-1), bins=100,)
    plt.title(w2_title, fontsize=titlefontsize)
    plt.tick_params(axis = 'x', labelsize =tickfontsize)
    plt.tick_params(axis = 'y', labelsize =tickfontsize)   
    plt.tight_layout()
    if filenames is not None:
        plt.savefig(filenames[1])
    plt.show()
