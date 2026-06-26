import numpy as np

"""
This module provides functions for various statistical purposes (especially in 
quantifying grain size/shape statistics from Digital Dark Field images made from
Machine Learning components).
"""


def mean_std(dataarray, weights=None):
    """
    Calculates weighted averages, with a sample correction for finite sample size
    from a much larger population

    Parameters
    ----------
    dataarray: np.ndarray
        array of values to be analysed
    weights: None or np.ndarray
        None gives unweighted values equal to np.mean() and np.std()
        array weights of same dimensions as dataarray gives weighted average
        and standard deviation
    Returns
    -------
    mean: float
    std: float
    """
    if isinstance (weights, np.ndarray):
        assert (
            weights.shape == dataarray.shape
        ), "weights must be the same shape as the data"
        mean = np.average(dataarray, weights=weights)
        std = np.sqrt(
            (weights * (dataarray - mean) ** 2).sum()
            / (weights.sum() - (weights**2).sum() / weights.sum())
        )
    else:
        mean = np.mean(dataarray)
        std = dataarray.std()
    return mean, std
