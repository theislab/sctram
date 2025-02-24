#!/usr/bin/env python3

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


from loguru import logger
_logger = logger.bind(name="BaseMetric")
try:
    from sctram.evaluate._metrics.validators import validate_between_minus_plus_1 as _validator
except ImportError:
    _logger.warning(f"Validation function not found. Skipping validation: {__file__}")
    def _validator(*args, **kwargs):
        pass


def pearson_correlation(given_pseudotime_array: np.ndarray,
                      inferred_pseudotime_array: np.ndarray,
                      validate_result: bool) -> float:
    """Compute Pearson correlation coefficient between two 1d arrays.

    Parameters:
        given_pseudotime_array (np.ndarray): A 1D array
        inferred_pseudotime_array (np.ndarray): A 1D array
        validate_result (bool): Whether to validate result with expected range.

    Returns:
        float: The Pearson correlation coefficient.

    Advantages:
        - Captures the degree of linear association between two connectivity structures.
        - Provides a global measure of similarity between the two graphs.

    Limitations:
        - Only measures linear relationships and may be sensitive to outliers.
        - Does not capture non-linear associations present in the graphs.

    Interpretation:
        - A value of 1 indicates perfect positive linear correlation.
        - A value of 0 suggests no linear correlation.
        - A value of -1 indicates perfect negative linear correlation.
    """    
    corr, _ = pearsonr(given_pseudotime_array, inferred_pseudotime_array)
    
    if validate_result:
        _validator(score=corr)
    
    return corr


def spearman_correlation(given_pseudotime_array: np.ndarray,
                       inferred_pseudotime_array: np.ndarray,
                       validate_result: bool) -> float:
    """Compute Spearman rank correlation coefficient between two 1d arrays.

    Parameters:
        given_pseudotime_array (np.ndarray): A 1D array
        inferred_pseudotime_array (np.ndarray): A 1D array
        validate_result (bool): Whether to validate result with expected range.

    Returns:
        float: The Spearman rank correlation coefficient.

    Advantages:
        - Assesses the monotonic relationship between the matrices regardless of the linearity.
        - Robust against non-normal distributions and outliers compared to Pearson correlation.

    Limitations:
        - Ignores the actual values in favor of rank ordering, potentially missing nuances in scale.
        - Sensitive to changes in the ordering of data points.

    Interpretation:
        - A value of 1 denotes a perfect monotonic increasing relationship.
        - A value of 0 denotes no monotonic relationship.
        - A value of -1 denotes a perfect monotonic decreasing relationship.
    """
    corr, _ = spearmanr(given_pseudotime_array, inferred_pseudotime_array)
    
    if validate_result:
        _validator(score=corr)
    
    return corr


def kendall_correlation(given_pseudotime_array: np.ndarray,
                      inferred_pseudotime_array: np.ndarray,
                      validate_result: bool) -> float:
    """Compute Kendall's tau correlation coefficient between 1d arrays.

    Parameters:
        given_pseudotime_array (np.ndarray): A 1D array
        inferred_pseudotime_array (np.ndarray): A 1D array
        validate_result (bool): Whether to validate result with expected range.

    Returns:
        float: The Kendall's tau correlation coefficient.

    Advantages:
        - Provides a non-parametric measure of rank correlation.
        - Robustly measures the probability that the orderings of the matrices are consistent.

    Limitations:
        - Computationally more intensive for large matrices.
        - Less sensitive than Pearson and Spearman to differences when many tied ranks are present.

    Interpretation:
        - A value of 1 indicates perfect agreement in the orderings.
        - A value of 0 indicates no association between the rankings.
        - A value of -1 indicates complete disagreement in the rankings.
    """
    tau, _ = kendalltau(given_pseudotime_array, inferred_pseudotime_array)
    
    if validate_result:
        _validator(score=tau)
    return tau
