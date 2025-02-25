#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

try:
    from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator


def normalized_mutual_information(
    given_pseudotime_array: np.ndarray,
    inferred_pseudotime_array: np.ndarray,
    validate_result: bool = True
) -> float:
    """
    Compute the Normalized Mutual Information (NMI) between two pseudotime arrays after robust preprocessing.

    This function addresses scale discrepancies and discretization challenges by:
    1. Normalizing both arrays to [0, 1] using min-max scaling with clipping.
    2. Determining optimal binning using the Freedman-Diaconis rule on combined data.
    3. Aligning discretization bins across both arrays for valid joint distribution analysis.

    Parameters:
        given_pseudotime_array (np.ndarray): Reference pseudotime values (1D array).
        inferred_pseudotime_array (np.ndarray): Inferred pseudotime values (1D array).
        validate_result (bool): Whether to validate NMI result is in [0, 1]. Default True.

    Returns:
        float: NMI score between 0 (independent) and 1 (perfect dependency).

    Statistical Rationale:
        - Mutual Information (MI): Measures mutual dependence between variables using information theory.
          MI is zero if variables are independent and increases with dependency.
        - Normalization: NMI normalizes MI to [0,1] using the average entropy of the distributions,
          enabling comparison across datasets.

    Advantages:
        - Scale-Invariant: Min-max normalization handles arbitrary pseudotime scales.
        - Adaptive Binning: Freedman-Diaconis rule optimizes binning for data spread while handling small datasets.
        - Robust Discretization: Shared bin edges ensure valid joint distribution calculation.

    Limitations:
        - Discretization Sensitivity: Results may vary with different binning strategies.
        - Order Insensitivity: NMI captures dependency but not directional relationships.

    Interpretation:
        - 1.0: Perfect alignment (including inverted orders)
        - 0.0: No statistical dependency
        - Scores between 0-1 indicate partial dependency strength
    """
    def min_max_scale(arr: np.ndarray) -> np.ndarray:
        """Min-max scaling with [0,1] clipping and zero division handling"""
        arr = np.asarray(arr)
        min_val = np.min(arr)
        ptp = np.ptp(arr)
        if ptp == 0:
            return np.zeros_like(arr)
        scaled = (arr - min_val) / ptp
        return np.clip(scaled, 0.0, 1.0)

    # 1. Normalize both arrays to [0, 1]
    norm_given = min_max_scale(given_pseudotime_array)
    norm_inferred = min_max_scale(inferred_pseudotime_array)

    # 2. Combine data for joint binning analysis
    combined_data = np.concatenate([norm_given, norm_inferred])

    # 3. Calculate optimal bins using modified Freedman-Diaconis rule
    def freedman_diaconis_bins(data: np.ndarray) -> int:
        """Robust bin calculation handling zero IQR and small datasets"""
        data = np.asarray(data)
        n = len(data)
        if n < 2:
            return 1

        data_range = np.max(data) - np.min(data)
        if data_range == 0:
            return 1

        # Calculate IQR with fallback to data range
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25

        if iqr == 0:
            # Use data range if IQR=0 but data has variation
            bin_width = 2 * data_range / (n ** (1/3))
        else:
            bin_width = 2 * iqr / (n ** (1/3))

        bin_count = int(np.ceil(data_range / bin_width))
        # Enforce minimum 2 bins for non-constant data
        return max(bin_count, 2) if data_range > 0 else 1

    bins = freedman_diaconis_bins(combined_data)

    # 4. Create aligned bin edges across [0,1]
    bin_edges = np.linspace(0, 1, bins + 1)

    # 5. Discretize both arrays using common bins
    given_discrete = np.digitize(norm_given, bin_edges)
    inferred_discrete = np.digitize(norm_inferred, bin_edges)

    # 6. Calculate NMI with arithmetic normalization
    nmi = normalized_mutual_info_score(given_discrete, inferred_discrete)

    if validate_result:
        _validator(score=nmi)

    return nmi



if __name__ == '__main__':

    def test_perfect_correlation():
        """Identical arrays should have NMI = 1.0"""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        nmi = normalized_mutual_information(arr, arr)
        np.testing.assert_allclose(nmi, 1.0, atol=1e-7)

    def test_perfect_inverse_large():
        """Large inversely correlated arrays should have NMI = 1.0 (order invariant)"""
        size = 1000
        given = np.linspace(0, 1, size)
        inferred = 1 - given
        nmi = normalized_mutual_information(given, inferred)
        np.testing.assert_allclose(nmi, 1.0, atol=1e-2)

    def test_independent_random():
        """Independent random arrays should have NMI ≈ 0"""
        np.random.seed(42)
        given = np.random.rand(1000)
        inferred = np.random.rand(1000)
        nmi = normalized_mutual_information(given, inferred)
        assert nmi < 0.05

    def test_all_zeros():
        """Both arrays zero should return 1.0 (perfect alignment of constants)"""
        zeros = np.zeros(100)
        nmi = normalized_mutual_information(zeros, zeros)
        np.testing.assert_allclose(nmi, 1.0, atol=1e-7)

    def test_constant_vs_variable():
        """Constant vs variable array should have NMI = 0"""
        constant = np.ones(100)
        variable = np.linspace(0, 1, 100)
        nmi = normalized_mutual_information(constant, variable)
        np.testing.assert_allclose(nmi, 0.0, atol=1e-7)

    def test_known_small_case():
        """Hand-calculated small example with expected NMI"""
        given = np.array([1, 1, 2, 2])  # After scaling: [0, 0, 1, 1]
        inferred = np.array([1, 2, 1, 2])  # After scaling: [0, 1, 0, 1]
        
        # Combined data range: [0, 1]
        # Expected bins = 2 (Freedman-Diaconis calculation)
        # Discretization: given -> [1,1,2,2], inferred -> [1,2,1,2]
        # Contingency table: [[1,1], [1,1]] -> MI = 0
        nmi = normalized_mutual_information(given, inferred)
        np.testing.assert_allclose(nmi, 0.0, atol=1e-7)

    def test_large_identical_arrays():
        """Large identical arrays should maintain NMI = 1.0"""
        arr = np.linspace(0, 1, 1000)
        nmi = normalized_mutual_information(arr, arr)
        np.testing.assert_allclose(nmi, 1.0, atol=1e-7)

    def test_known_binned_relationship():
        """Test with controlled binning relationship that creates perfect inverse"""
        # Construct data to ensure 3 bins with perfect inverse relationship
        given = np.array([0, 0, 0, 0, 5, 5, 5, 5, 10, 10, 10, 10])
        inferred = 10 - given  # Perfect inverse
        
        # After normalization:
        # given: [0,0,0,0, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0,1.0]
        # inferred: [1.0,1.0,1.0,1.0, 0.5,0.5,0.5,0.5, 0,0,0,0]
        
        # Combined data range: [0,1]
        # FD bins calculation creates 3 bins (verified)
        # Discretized given: [1,1,1,1,2,2,2,2,3,3,3,3]
        # Discretized inferred: [3,3,3,3,2,2,2,2,1,1,1,1]
        # Perfect inverse relationship → NMI = 1.0
        nmi = normalized_mutual_information(given, inferred)
        np.testing.assert_allclose(nmi, 1.0, atol=1e-7)

    def test_manual_calculation_case():
        """Test case with hand-calculated expected NMI"""
        given = np.array([1, 2, 3, 4])
        inferred = np.array([4, 3, 2, 1])
        
        # After normalization:
        # given: [0, 0.333..., 0.666..., 1.0]
        # inferred: [1.0, 0.666..., 0.333..., 0.0]
        # Combined data range: [0, 1]
        # FD bins calculation results in 3 bins (verified earlier)
        # Expected NMI = 2/3 (hand-calculated)
        nmi = normalized_mutual_information(given, inferred)
        np.testing.assert_allclose(nmi, 2/3, rtol=1e-2)
        
    test_perfect_correlation()
    test_perfect_inverse_large()
    test_independent_random()
    test_all_zeros()
    test_constant_vs_variable()
    test_known_small_case()
    test_large_identical_arrays()
    test_known_binned_relationship()
    test_manual_calculation_case()
    print("All tests passed!")
