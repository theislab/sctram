#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score

try:
    from sctram.evaluate._metrics._src.validators import validate_maximum_1 as _validator
except ImportError:
    from validators import validate_maximum_1 as _validator


def r_squared(given_pseudotime_array: np.ndarray,
                 inferred_pseudotime_array: np.ndarray,
                 validate_result: bool) -> float:
    """Calculate R-squared between two 1D arrays.
    
    This method computes the coefficient of determination (R-squared), which represents the 
    proportion of variance in the given (reference) pseudotime that is predictable from the 
    inferred pseudotime. This is a common metric in regression analysis for evaluating goodness-of-fit.
    
    Parameters:
        given_pseudotime_array (np.ndarray): A 1D array
        inferred_pseudotime_array (np.ndarray): A 1D array
        validate_result (bool): Whether to validate result with expected range.
        
    Returns:
        float: The R-squared value, ranging from negative infinity to 1, with higher values indicating a better fit.
        
    Advantages:
        - Provides an intuitive measure of how well the inferred matrix explains the variance in the reference matrix.
        - Widely used and understood in regression contexts.
        
    Limitations:
        - Sensitive to the data's range and distribution.
        - Can be misleading if non-linear relationships dominate.
        
    Interpretation:
        - An R-squared of 1 indicates a perfect fit.
        - A value of 0 indicates that the model explains none of the variance.
        - Negative values suggest that the model performs worse than a simple horizontal line.
    """
    r2 = r2_score(given_pseudotime_array, inferred_pseudotime_array)
    
    if validate_result:
        _validator(score=r2)
    
    return r2

def r_squared_with_spline(given_pseudotime_array: np.ndarray,
                             inferred_pseudotime_array: np.ndarray,
                             validate_result: bool,
                             smoothing_factor: float = None,
                             spline_degree: int = 3) -> float:
    """Calculate R-squared using a Univariate Spline fit between two 1D arrays.
    
    This method first sorts the inferred pseudotime and its corresponding reference values, fits a Univariate Spline 
    (of a specified degree) to capture potential non-linear relationships, and then computes the R-squared value 
    between the given pseudotime and the spline predictions.
    
    Parameters:
        given_pseudotime_array (np.ndarray): A 1D array
        inferred_pseudotime_array (np.ndarray): A 1D array
        validate_result (bool): Whether to validate result with expected range.
        spline_degree (int, optional): Degree of the smoothing spline. Defaults to 3.
        smoothing_factor (float, optional): Regularization parameter controlling smoothness. 
                                            Higher values increase smoothing. If None, uses 
                                            scipy's default (len(x)). Defaults to None.
        
    Returns:
        float: The R-squared value for the spline fit, indicating the goodness-of-fit for capturing non-linear relationships.
        
    Advantages:
        - Captures non-linear associations between the two matrices.
        - Offers a more flexible fit compared to a purely linear model.
        
    Limitations:
        - The result is sensitive to the choice of spline degree and the sorting procedure.
        - May overfit if the spline degree is set too high.
        
    Interpretation:
        - An R-squared value near 1 indicates an excellent non-linear fit.
        - Values around 0 suggest that the spline does not capture the variance well.
        - Negative values indicate a poor fit, potentially worse than a horizontal line.
    """
    sorted_indices = np.argsort(inferred_pseudotime_array)
    x_sorted = inferred_pseudotime_array[sorted_indices]
    y_sorted = given_pseudotime_array[sorted_indices]
    
    # Fit the Univariate Spline to the sorted data
    spline = UnivariateSpline(x_sorted, y_sorted, k=spline_degree, s=smoothing_factor)
    # Predict y values using the original inferred values
    y_pred = spline(inferred_pseudotime_array)
    r2 = r2_score(given_pseudotime_array, y_pred)
    
    if validate_result:
        _validator(score=r2)
        
    return r2


    # y = given_pseudotime
    # x = inferred_pseudotime
    # y_mean = np.mean(y)
    # ss_total = np.sum((y - y_mean)**2)
    # if np.isclose(ss_total, 0):
    #     _logger.warning("Given pseudotime is constant. R² is undefined.")
    #     return 0.0
    
    # Cross-validation setup
    # kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # s_values = np.logspace(-6, 0, num=20) * ss_total  # s candidates scaled by SS_total
    # best_s, best_mse = None, np.inf

    # for s in s_values:
    #     mse_fold = []
    #     for train_idx, test_idx in kf.split(x):
    #         x_train, y_train = x[train_idx], y[train_idx]
    #         x_test, y_test = x[test_idx], y[test_idx]

    #         # Preprocess training data: sort and aggregate duplicates
    #         sorted_idx = np.argsort(x_train)
    #         x_sorted = x_train[sorted_idx]
    #         y_sorted = y_train[sorted_idx]
    #         x_unique, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    #         if len(x_unique) < 2:
    #             continue  # Insufficient unique x to fit spline
    #         y_unique = np.array([np.mean(y_sorted[inverse == i]) for i in range(len(x_unique))])

    #         try:
    #             spline = UnivariateSpline(x_unique, y_unique, s=s, check_finite=True)
    #             y_pred = spline(x_test)
    #             mse = np.mean((y_test - y_pred)**2)
    #             mse_fold.append(mse)
    #         except Exception as e:
    #             _logger.debug(f"Spline fit failed for s={s:.2e}: {str(e)}")
    #             continue
        
    #     if len(mse_fold) == cv_folds:  # Require successful fits in all folds
    #         avg_mse = np.mean(mse_fold)
    #         if avg_mse < best_mse:
    #             best_mse, best_s = avg_mse, s

    # if best_s is None:
    #     _logger.warning("No valid spline fit found. Returning R²=0.")
    #     return 0.0

    # Final model fit on entire dataset
    # sorted_idx = np.argsort(x)
    # x_sorted = x[sorted_idx]
    # y_sorted = y[sorted_idx]
    # x_unique, inverse = np.unique(x_sorted, return_inverse=True)
    # if len(x_unique) < 2:
    #     return 0.0
    # y_unique = np.array([np.mean(y_sorted[inverse == i]) for i in range(len(x_unique))])
    # spline = UnivariateSpline(x_unique, y_unique, s=best_s)
    # y_pred = spline(x)
    # ss_res = np.sum((y - y_pred)**2)
    # r_squared = 1.0 - (ss_res / ss_total)


if __name__ == "__main__":
    
    def test_perfect_linear_relationship_shuffled():
        np.random.seed(42)
        inferred = np.arange(100)
        np.random.shuffle(inferred)
        given = 2.5 * inferred + 3.0
        r2 = r_squared_with_spline(given, inferred, validate_result=False, spline_degree=1)
        assert np.isclose(r2, 1.0), f"Expected R²=1.0, got {r2}"

    def test_perfect_cubic_relationship():
        inferred = np.linspace(-10, 10, 1000)
        given = inferred ** 3
        r2 = r_squared_with_spline(given, inferred, validate_result=False, spline_degree=3)
        assert np.isclose(r2, 1.0), f"Expected R²=1.0, got {r2}"

    def test_constant_given():
        np.random.seed(43)
        inferred = np.random.rand(100)
        given = np.full(100, 5.0)
        r2 = r_squared_with_spline(given, inferred, validate_result=False, spline_degree=3)
        assert np.isclose(r2, 0.0), f"Expected R²=0.0, got {r2}"

    def test_inverse_linear_relationship():
        inferred = np.arange(100)
        given = 100 - inferred
        r2 = r_squared_with_spline(given, inferred, validate_result=False, spline_degree=1)
        assert np.isclose(r2, 1.0), f"Expected R²=1.0, got {r2}"

    def test_large_random_noise():
        np.random.seed(44)
        inferred = np.random.rand(1000)
        given = np.random.rand(1000)
        r2 = r_squared_with_spline(given, inferred, validate_result=False, spline_degree=3)
        assert r2 < 0.5, f"Expected low R² for random noise, got {r2}"

    def test_manual_calculation_spline_degree_1():
        inferred = np.array([1.0, 2.0, 3.0])
        given = np.array([1.0, 3.0, 6.0])
        # Manually calculate expected R²
        sorted_indices = np.argsort(inferred)
        x_sorted = inferred[sorted_indices]
        y_sorted = given[sorted_indices]
        spline = UnivariateSpline(x_sorted, y_sorted, k=1, s=len(x_sorted))
        y_pred = spline(inferred)
        expected_r2 = r2_score(given, y_pred)
        r2 = r_squared_with_spline(given, inferred, validate_result=False, spline_degree=1)
        assert np.isclose(r2, expected_r2, atol=1e-6), f"Expected R²={expected_r2}, got {r2}"

    def test_perfect_quadratic_relationship_degree_2():
        inferred = np.linspace(-5, 5, 500)
        given = inferred ** 2
        r2 = r_squared_with_spline(given, inferred, validate_result=False, spline_degree=2)
        assert np.isclose(r2, 1.0), f"Expected R²=1.0, got {r2}"
    
    test_perfect_linear_relationship_shuffled()
    test_perfect_cubic_relationship()
    test_constant_given()
    test_inverse_linear_relationship()
    test_large_random_noise()
    test_manual_calculation_spline_degree_1()
    test_perfect_quadratic_relationship_degree_2()
    print("All tests passed!")
