import numpy as np
from dmd import hankel, dmd, unroll
from inference import AdaptiveEWMA
from criterion import mean_absolute_increments

def single_change(data, burn_in, window, order, rank, gamma, ell):
    """
    Detect the first changepoint in the data stream using adaptive EWMA algorithm
    on increments of the reconstruction error.

    Parameters
    ----------
    data : array_like of shape (num_features, sequence_length)
        A real or complex array, with time as columns and dimensions along rows.
    burn_in : int
        Number of initial observations in the sequence to be ignored.
    window : int
        Size of the sliding window.
    order : int
        Autoregressive order for the number of time-delay embeddings.
    rank : int
        Rank for the DMD truncation.
    gamma : float
        Learning rate for the adaptive EWMA.
    ell : float
        Control limit for the adaptive EWMA.

    Returns
    -------
    changepoint : int or None
        Time of first changepoint detected in the data, or None if no changepoint is detected.
    """
    # initalisation
    num_features, sequence_length = data.shape
    ewma = AdaptiveEWMA(gamma, ell, burn_in-window)
    previous_epsilon = 0
    # sequential processing
    for t in range(window, sequence_length):
        X = data[:, t-window:t] # window batch
        H = hankel(X, order) # hankel batch
        modes, dynamics_evolution, amplitudes = dmd(H, rank) # dmd
        H_hat = modes @ amplitudes @ dynamics_evolution # reconstructed hankel batch
        X_hat = unroll(H_hat, order) # reconstructed window batch
        # error, increment
        epsilon = np.linalg.norm(X - X_hat) ** 2 / (num_features * window)
        delta = epsilon - previous_epsilon
        previous_epsilon = epsilon
        # adaptive ewma
        if t <= window + 1:
            previous_delta = delta
        elif t == window + 2:
            ewma.initialise_statistics([previous_delta, delta])
        else:
            ewma.update_statistics(delta)
            ewma.update_mean_variance(delta)
            if t >= burn_in:
                ewma.decision_rule()
                if ewma.detected_change:
                    return max(t-1, burn_in)
    return None

def multiple_changes(data, burn_in, gamma, ell, optimiser, stop_at_first=False):
    """
    Detect every changepoint in the data stream using adaptive EWMA algorithm
    on increments of the reconstruction error.
    This implementation is auto-adaptive: after each detected change, the 
    hyperparameters are reset using a new burn-in period.

    Parameters
    ----------
    data : array_like of shape (num_features, sequence_length)
        A real or complex array, with time as columns and dimensions along rows.
    burn_in : int
        Number of initial observations in the sequence to be ignored.
    gamma : float
        Learning rate for the adaptive EWMA.
    ell : float
        Control limit for the adaptive EWMA.
    optimiser : function
        A function of the form optimiser(objective, data) to find the optimal parameters.
    stop_at_first : bool
        If True, the function stops after the first changepoint is detected.

    Returns
    -------
    changepoints : list
        Times of detected changepoints in the data stream.
    """
    # initalisation
    num_features, sequence_length = data.shape
    best_params = optimiser(mean_absolute_increments, data[:, :burn_in])
    tau = single_change(data, burn_in, **best_params, gamma=gamma, ell=ell)
    if tau is None:
        return []
    elif stop_at_first:
        return [tau]
    changepoints = [tau]
    # sequential processing
    while (tau is not None) and (tau + burn_in < sequence_length):
        # hyperparameter selection
        best_params = optimiser(mean_absolute_increments, data[:, tau:tau+burn_in])
        # next changepoint detection
        tau = single_change(data[:, tau:], burn_in, **best_params, gamma=gamma, ell=ell)
        # if detected change, store and update
        if tau:
            tau += changepoints[-1]
            changepoints.append(tau)
    return changepoints