import numpy as np

def check_parameters(burn_in, window, order, rank):
    """
    Check the validity of the parameters.
    
    Parameters
    ----------
    burn_in : int
        Number of initial observations in the sequence to be ignored.
    window : int
        Size of the sliding window.
    order : int
        Autoregressive order for the number of time-delay embeddings.
    rank : int
        Rank of the truncated SVD decomposition.
    
    Returns
    ------
    valid
        True if the parameters are valid, False otherwise.
    """
    valid = rank < order < window < burn_in
    return valid

def grid_search(objective, data, grid):
    """
    Grid search for the optimal parameters of the objective function.
    
    Parameters
    ----------
    objective : function
        A function of the form objective(data, window, order, rank) to be minimised.
    data : array_like of shape (num_features, burn_in)
        A real or complex array, with time as columns and dimensions along rows, which 
        corresponds to the burn-in period.
    grid : dict
        A dictionary containing the grid search parameters.
    
    Returns
    -------
    best_params : dict
        A dictionary containing the optimal parameters.
    """
    # initialisation
    num_features, burn_in = data.shape
    best_params = {}
    best_score = np.inf
    # grid search
    for window in grid["window"]:
        for order in grid["order"]:
            for rank in grid["rank"]:
                # check parameters validity
                valid = check_parameters(burn_in, window, order, rank)
                if not valid:
                    continue
                # evaluate the objective function
                score = objective(data, window, order, rank)
                # update the best parameters
                if score < best_score:
                    best_score = score
                    best_params = {"window": window, "order": order, "rank": rank}
    return best_params