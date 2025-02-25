#!/usr/bin/env python3

import numpy as np
from scipy.stats import cramervonmises_2samp, ks_2samp
from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator1
from sctram.evaluate._metrics.validators import validate_zero_or_positive as _validator2


def cdf_cramer_von_mises(given_pseudotime_array: np.ndarray,
                     inferred_pseudotime_array: np.ndarray,
                     validate_result: bool) -> float:
    """Compute the Cramér-von Mises (CvM) statistic between two 1D arrays.

    This function quantifies the overall discrepancy between two distributions by calculating the integrated
    squared differences between their empirical cumulative distribution functions (CDFs) using the CvM test.

    Parameters:
        given_array (np.ndarray): A 1D array representing the given distribution.
        inferred_array (np.ndarray): A 1D array representing the inferred distribution.
        validate_result (bool): Whether to validate the computed CvM statistic using the expected range.

    Returns:
        float: The CvM statistic reflecting the overall difference between the two distributions.

    Advantages:
        - Comprehensive: Considers discrepancies over the entire range of the data.
        - Non-parametric: Does not assume any specific distribution form.

    Limitations:
        - Sensitivity: May be influenced by outliers or extreme values.
        - Sample size: Requires a sufficiently large number of observations for stable estimation.

    Interpretation:
        - A higher CvM statistic indicates a greater overall difference between the cumulative distributions.
    """
    cvm_stat = cramervonmises_2samp(inferred_pseudotime_array, given_pseudotime_array).statistic
    
    if validate_result:
        _validator2(score=cvm_stat)
    
    return cvm_stat


def cdf_kolmogorov_smirnov(given_pseudotime_array: np.ndarray,
                                  inferred_pseudotime_array: np.ndarray,
                                  validate_result: bool) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic between two 1D arrays.

    This function compares two distributions by calculating the maximum absolute difference
    between their empirical cumulative distribution functions (CDFs) using the KS test.

    Parameters:
        given_array (np.ndarray): A 1D array representing the given distribution.
        inferred_array (np.ndarray): A 1D array representing the inferred distribution.
        validate_result (bool): Whether to validate the computed KS statistic using the expected range.

    Returns:
        float: The KS statistic indicating the maximum discrepancy between the two CDFs.

    Advantages:
        - Non-parametric: Does not require any assumptions about the underlying distributions.
        - Sensitive: Detects the maximum point-wise difference between the two distributions.

    Limitations:
        - Sensitivity: Tends to be more responsive around the median than in the tails.
        - Sample size: Its reliability can decrease with very small datasets.

    Interpretation:
        - A higher KS statistic implies a more significant difference between the two distributions.
    """
    ks_statistic, _ = ks_2samp(inferred_pseudotime_array, given_pseudotime_array)
    
    if validate_result:
        _validator1(score=ks_statistic)
    
    return ks_statistic