#!/usr/bin/env python3

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wd

from sctram.evaluate._metrics._src.utils import prepare_pseudotime
from sctram.evaluate._metrics._src.validators import validate_zero_or_positive as _validator


def wasserstein_distance_pseudotime(
    given_pseudotime_array: np.ndarray, inferred_pseudotime_array: np.ndarray, validate_result: bool
) -> float:
    """Compute the Wasserstein distance between two 1D arrays.

    This method computes the first Wasserstein distance (Earth Mover's Distance) between two probability distributions.
    It quantifies the minimum "cost" required to transform one distribution into another, where cost is defined as the
    product of the amount of probability mass moved and the distance by which it is moved.

    Parameters:
        given_pseudotime_array (np.ndarray): A 1D array representing the reference pseudotime distribution.
        inferred_pseudotime_array (np.ndarray): A 1D array representing the inferred pseudotime distribution.
        validate_result (bool): Whether to validate that the computed distance lies within the expected range.

    Returns:
        float: The computed Wasserstein distance.

    Advantages:
        - Measures the overall dissimilarity between two probability distributions.
        - Captures both location shifts and dispersion differences between the distributions.

    Limitations:
        - Sensitive to differences in both the location and spread of the distributions.
        - May not capture fine-grained local differences in the distribution shapes.

    Interpretation:
        - A value of 0 indicates that the two distributions are identical.
        - Higher values indicate greater dissimilarity between the distributions.
    """
    # Measure absolute distances. Normalization focuses on shape rather than magnitude.
    norm_given = prepare_pseudotime(given_pseudotime_array, method="minmax")
    norm_inferred = prepare_pseudotime(inferred_pseudotime_array, method="minmax")
    # TODO: fix here.
    wd = scipy_wd(given_pseudotime_array, inferred_pseudotime_array)

    if validate_result:
        _validator(score=wd)

    return wd
