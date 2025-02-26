#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sctram.evaluate._metrics._src.validators import validate_zero_or_positive as _validator
from sctram.evaluate._metrics._src.utils import prepare_pseudotime


def mse(given_pseudotime_array: np.ndarray,
        inferred_pseudotime_array: np.ndarray,
        validate_result: bool) -> float:
    """Compute the Mean Squared Error (MSE) between two 1D arrays.
    
    Parameters:
        given_pseudotime_array (np.ndarray): A 1D array
        inferred_pseudotime_array (np.ndarray): A 1D array
        validate_result (bool): Whether to validate result with expected range.

    Returns:
        float: The Mean Squared Error (MSE).
        
    Advantages:
        - Penalizes larger errors more than smaller ones due to the squaring effect.
        - Provides a clear quantitative measure of overall error magnitude.
        
    Limitations:
        - Highly sensitive to outliers because errors are squared.
        - Requires both matrices to be on the same scale.
        
    Interpretation:
        - Lower MSE values indicate better agreement between the given and inferred matrices.
    """
    given_pseudotime_array = prepare_pseudotime(given_pseudotime_array, method="minmax")
    inferred_pseudotime_array = prepare_pseudotime(inferred_pseudotime_array, method="minmax")
    
    mse_value = mean_squared_error(given_pseudotime_array, inferred_pseudotime_array)
    
    if validate_result:
        _validator(score=mse_value)
    
    return mse_value


def mae(given_pseudotime_array: np.ndarray,
        inferred_pseudotime_array: np.ndarray,
        validate_result: bool) -> float:
    """Compute the Mean Absolute Error (MAE) between two 1D arrays.
    
    Parameters:
        given_pseudotime_array (np.ndarray): A 1D array
        inferred_pseudotime_array (np.ndarray): A 1D array
        validate_result (bool): Whether to validate result with expected range.
        
    Returns:
        float: The Mean Absolute Error (MAE).
        
    Advantages:
        - Provides a linear measure that does not exaggerate larger errors.
        - More robust to outliers compared to MSE.
        
    Limitations:
        - Does not penalize larger errors as strongly as the MSE.
        - Requires both matrices to be on the same scale.
        
    Interpretation:
        - Lower MAE values indicate better agreement between the given and inferred matrices.
    """
    given_pseudotime_array = prepare_pseudotime(given_pseudotime_array, method="minmax")
    inferred_pseudotime_array = prepare_pseudotime(inferred_pseudotime_array, method="minmax")
    
    mae_value = mean_absolute_error(given_pseudotime_array, inferred_pseudotime_array)
    
    if validate_result:
        _validator(score=mae_value)
    
    return mae_value
