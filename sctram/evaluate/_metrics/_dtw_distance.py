#!/usr/bin/env python3

import numpy as np
import fastdtw

try:
    from sctram.evaluate._metrics.validators import validate_zero_or_positive as _validator
except ImportError:
    from validators import validate_zero_or_positive as _validator


def dtw_distance(given_pseudotime_array: np.ndarray,
                             inferred_pseudotime_array: np.ndarray,
                             validate_result: bool,
                             radius: int = None):
    """Compute the normalized Dynamic Time Warping (DTW) distance between two 1D arrays of pseudotime.

    This method employs DTW to capture
    similarities between sequences that may vary in timing or speed. It determines the optimal
    alignment by warping the time axis, ensuring that sequences with local shifts or variations
    can be compared effectively.

    Parameters: 
        given_pseudotime_array (np.ndarray): A 1D array
        inferred_pseudotime_array (np.ndarray): A 1D array
        radius (int): Search radius for FastDTW. Automatically determined if None. Default: None.
        validate_result (bool): Whether to validate result with expected range.

    Returns:
        float: The normalized DTW distance is defined as the cumulative DTW distance divided by the
            length of the optimal warping path.

    Advantages:
        - Captures similarities between sequences even when they are not perfectly aligned in time.
        - Provides an optimal alignment by warping the time axis, which can accommodate variable speeds
        or local shifts.
        - Normalization by the path length allows for a scale-independent comparison.

    Limitations:
        - Sensitive to the shape and local variations of the sequences, which may affect the DTW distance.
        - Can be computationally intensive for long sequences, although using the 'fastdtw' library
        alleviates this to some extent.
        - Assumes that both input arrays are one-dimensional and of comparable length or structure;
        otherwise, pre-processing may be required.

    Interpretation:
        A lower normalized DTW distance indicates a better alignment between the inferred and reference
        pseudotime sequences, suggesting that the inferred ordering closely matches the reference ordering.
        Conversely, a higher value implies greater dissimilarity between the sequences.
    """
    # Set adaptive radius if not specified (heuristic: 0.1% of length)
    if radius is None:
        radius = max(1, int(np.ceil(len(given_pseudotime_array) / 1e3)))
    
    distance, path = fastdtw.fastdtw(
        given_pseudotime_array,
        inferred_pseudotime_array,
        radius=radius
    )
    
    # Normalize by path length to remove scale dependence
    path_length = len(path)
    score = distance / path_length if path_length > 0 else 0.0
    
    if validate_result:
        _validator(score=score)
        
    return score


if __name__ == "__main__":
        
    def test_identical_arrays():
        given = np.array([1.0, 2.0, 3.0])
        inferred = np.array([1.0, 2.0, 3.0])
        score = dtw_distance(given, inferred, validate_result=False)
        assert np.isclose(score, 0.0), f"Expected 0.0, got {score}"

    def test_shifted_arrays():
        given = np.array([1, 2, 3])
        inferred = given + 2  # Shifted by 2 units
        
        radius = max(1, int(np.ceil(len(given) / 1e3)))  # Match metric's logic
        distance, path = fastdtw.fastdtw(given, inferred, radius=radius)
        expected_score = distance / len(path)
        
        actual_score = dtw_distance(given, inferred, validate_result=False)
        
        assert np.isclose(actual_score, expected_score), \
            f"Score mismatch. Expected {expected_score}, got {actual_score}"

    def test_reversed_arrays():
        given = np.array([1, 2, 3])
        inferred = np.array([3, 2, 1])
        score = dtw_distance(given, inferred, validate_result=False)
        expected = 4.0 / 3.0
        assert np.isclose(score, expected, rtol=1e-4), f"Expected {expected}, got {score}"

    def test_constant_arrays():
        given = np.array([5.0, 5.0, 5.0])
        inferred = np.array([5.0, 5.0, 5.0])
        score = dtw_distance(given, inferred, validate_result=False)
        assert np.isclose(score, 0.0), f"Expected 0.0, got {score}"

    def test_large_arrays():
        size = 1000
        given = np.zeros(size)
        inferred = np.ones(size)
        score = dtw_distance(given, inferred, validate_result=False)
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def test_noisy_arrays():
        given = np.array([1.0, 2.0, 3.0])
        noise = np.array([0.1, -0.1, 0.0])
        inferred = given + noise
        expected = abs(noise).sum() / len(given)  # Using Manhattan distance (1D Euclidean = absolute)
        score = dtw_distance(given, inferred, validate_result=False)
        assert np.isclose(score, expected), f"Expected {expected}, got {score}"

    def test_single_element():
        given = np.array([5])
        inferred = np.array([7])
        score = dtw_distance(given, inferred, validate_result=False)
        assert np.isclose(score, 2.0), f"Expected 2.0, got {score}"

    def test_phase_shifted_sine_wave():
        # Generate two sine waves with a phase shift
        t = np.linspace(0, 2*np.pi, 1000)
        phase_shift = np.pi/2  # 90-degree phase shift
        given = np.sin(t)
        inferred = np.sin(t + phase_shift)
        
        # Expected DTW distance calculated empirically for validation
        # Manually check that DTW aligns the shifted waves correctly
        score = dtw_distance(given, inferred, validate_result=False)
        # Since DTW should handle phase shifts, the distance should be relatively low
        # Empirical check: actual value should be consistent across runs
        assert score < 0.5, f"DTW score unexpectedly high for phase-shifted sines: {score}"
    
    test_identical_arrays()
    test_shifted_arrays()
    test_reversed_arrays()
    test_constant_arrays()
    test_large_arrays()
    test_noisy_arrays()
    test_single_element()
    test_phase_shifted_sine_wave()
    print("All tests passed!")
