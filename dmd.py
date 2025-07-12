import numpy as np
from scipy.linalg import svd

def dmd(data, rank):
    """
    Fit Dynamic Mode Decomposition algorithm to input data, learning the dominant 
    modes and dynamics of the system.

    Parameters
    ----------
    data : array_like, shape (num_features, sequence_length)
        A real or complex array, with time steps as columns and dimensions along rows.
    rank : int
        The singular value decomposition truncation rank to be used for the algorithm.

    Returns
    -------
    reconstructed_data : array_like, shape (num_features, sequence_length)
        A real or complex array of the reconstructed input stream X. 
    """
    # lagged snapshots
    X, Y = data[:, :-1], data[:, 1:]
    # truncated svd of X
    try:
        U, S, V_t = svd(X, full_matrices=False, lapack_driver="gesdd") 
    except Exception:
        U, S, V_t = svd(X, full_matrices=False, lapack_driver="gesvd")
    U = U[:, :rank]
    S = S[:rank]
    V_t = V_t[:rank, :]
    # reduced linear operator
    A = U.conj().T @ Y @ V_t.conj().T @ np.diag(1 / S)
    # modes, dynamics, amplitudes
    dynamics, W = np.linalg.eig(A)
    modes = Y @ V_t.conj().T @ np.diag(1 / S) @ W
    amplitudes = np.diag(np.linalg.pinv(modes) @ X[:, 0])
    dynamics_evolution = np.array([dynamics ** t for t in range(Y.shape[1] + 1)]).T
    return modes, dynamics_evolution, amplitudes

def hankel(data, order):
    """
    Compute the Hankel matrix for the given array with given autoregressive order.

    Parameters
    ----------
    data : array_like of shape (num_features, sequence_length)
        A real or complex array, with time as columns and dimensions along rows.
    order : int
        Autoregressive order.
    
    Returns
    -------
    H : array_like of shape (num_features * order, sequence_length - order + 1)
        A real or complex array, corresponding to the full autoregressive form of
        input with given autoregressive order. For multivariate inputs, the 
        autoregressive forms are computed separately for each component and 
        stacked.
    """
    num_features, sequence_length = data.shape
    # hankel block matrices for each feature
    sub_arrays = [
        np.array([data[i, j:j+order] for j in range(sequence_length - order + 1)]).T
        for i in range(num_features)
    ]
    # stack the hankel block matrices
    H = np.concatenate(sub_arrays)
    return H

def unroll(H, order):
    """
    Unroll the Hankel matrix to the original observation space.

    Parameters
    ----------
    H : array_like of shape (num_features * order, sequence_length - order + 1)
        A real or complex array, corresponding to the full autoregressive form of
        input with given autoregressive order.
    order : int
        Autoregressive order.

    Returns
    -------
    data : array_like of shape (num_features, sequence_length)
        A real or complex array, corresponding to the original observation space.
    """
    num_features = H.shape[0] // order
    sequence_length = H.shape[1] + order - 1
    # slice the hankel matrix into blocks
    sub_arrays = [
        np.array(H[i*order:(i+1)*order, :]) 
        for i in range(num_features)
    ]
    # unroll the hankel blocks
    data = np.array([
        np.concatenate([sub_arrays[i][0, :], sub_arrays[i][1:, -1]])
        for i in range(num_features)
    ])
    return data