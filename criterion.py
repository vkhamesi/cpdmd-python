import numpy as np
from dmd import hankel, dmd, unroll

def mean_absolute_increments(data, window, order, rank):
    """
    Compute a score based on the reconstruction error for the given data, window size 
    and autoregressive order, on the burn-in period. 
    The score is the mean of absolute values of increments values on the burn-in period.
    Considered as an objective function used as input for the optimisation algorithm.
    
    Parameters
    ----------
    data : array_like of shape (num_features, burn_in)
        A real or complex array, with time as columns and dimensions along rows, which 
        corresponds to the burn-in period.
    window : int
        Size of the sliding window.
    order : int
        Autoregressive order for the number of time-delay embeddings.
    rank : int
        Rank of the truncated SVD decomposition.
    
    Returns
    -------
    score : float
        The reconstruction error on the burn-in period.
    """
    # initalisation
    num_features, burn_in = data.shape
    delta_list = []
    prev_epsilon = 0
    # sequential processing
    for t in range(window, burn_in):
        X = data[:, t-window:t] # window batch
        H = hankel(X, order) # hankel batch
        modes, dynamics_evolution, amplitudes = dmd(H, rank) # dmd
        H_hat = modes @ amplitudes @ dynamics_evolution # reconstructed hankel batch
        X_hat = unroll(H_hat, order) # reconstructed window batch
        # error, increment and storage
        epsilon = np.linalg.norm(X - X_hat) ** 2 / (num_features * window)
        delta = epsilon - prev_epsilon
        prev_epsilon = epsilon
        delta_list.append(delta)
    # score computation
    delta_list = delta_list[1:]
    score = np.mean(window * np.abs(delta_list))
    return score