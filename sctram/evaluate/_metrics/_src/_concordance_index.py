#!/usr/bin/env python3

import numpy as np
from scipy.stats import rankdata

try:
    from sctram.evaluate._metrics._src.validators import validate_inclusive_between_0_1 as _validator
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator


def concordance_index(
    given_pseudotime_array: np.ndarray,
    inferred_pseudotime_array: np.ndarray,
    validate_result: bool = True
) -> float:
    """Compute the Concordance Index (CI) between two pseudotime arrays.

    The CI measures the probability that the order of pseudotimes in `inferred_pseudotime`
    matches the order in `given_pseudotime`. Ties in `given_pseudotime` are excluded, while
    ties in `inferred_pseudotime` contribute 0.5 to the concordance probability.

    Parameters:
        given_pseudotime (np.ndarray): 1D array of reference pseudotimes.
        inferred_pseudotime (np.ndarray): 1D array of inferred pseudotimes.
        validate_result (bool): Validate CI is within [0, 1]. Default True.

    Returns:
        float: CI value between 0 (no concordance) and 1 (perfect concordance).

    Raises:
        ValueError: If inputs are invalid or no admissible pairs exist.

    Statistical Notes:
        1. CI = (concordant + 0.5 * tied_inferred) / admissible_pairs
        2. Pairs with tied `given_pseudotime` are excluded (non-admissible).
        3. CI ranges [0,1], with 0.5 indicating random agreement.
        4. Mathematically equivalent to the area under the ROC curve for pairwise comparisons.
    
    Advantages:
        - Provides a global measure of agreement between two orderings.
        - Non-parametric and does not assume a specific relationship form between the arrays.
    
    Limitations:
        - Requires at least two samples for computation.
        - Sensitive to tied pairs; pairs with equal values in either array are excluded.
        - May be computationally intensive for very large arrays due to pairwise comparisons.
    
    Interpretation:
        - A value of 1 indicates perfect concordance (complete agreement in order).
        - A value of 0.5 suggests random concordance (no better than chance).
        - A value closer to 0 indicates poor agreement.
    """
    n = len(given_pseudotime_array)
    if n < 2:
        raise ValueError("At least two samples are required.")

    # Calculate pairwise differences using vectorized operations
    delta_t = given_pseudotime_array[:, None] - given_pseudotime_array[None, :]
    delta_p = inferred_pseudotime_array[:, None] - inferred_pseudotime_array[None, :]

    # Extract upper triangle indices (i < j)
    rows, cols = np.triu_indices(n, k=1)
    delta_t_upper = delta_t[rows, cols]
    delta_p_upper = delta_p[rows, cols]

    # Identify admissible pairs (non-zero delta_t)
    admissible_mask = delta_t_upper != 0
    delta_t_admissible = delta_t_upper[admissible_mask]
    delta_p_admissible = delta_p_upper[admissible_mask]

    if len(delta_t_admissible) == 0:
        raise ValueError("No admissible pairs (all given pseudotimes are tied).")

    # Calculate concordance components
    product = delta_t_admissible * delta_p_admissible
    concordant = (product > 0).sum()  # Same direction
    tied_p = (delta_p_admissible == 0).sum()  # Ties in inferred pseudotime
    total_concordant = concordant + 0.5 * tied_p
    ci = total_concordant / len(delta_t_admissible)

    if validate_result:
        _validator(score=ci)

    return ci


if __name__ == "__main__":
    def test_perfect_concordance():
        given = np.array([1, 2, 3, 4, 5])
        inferred = np.array([0.9, 2.1, 3.0, 3.9, 5.1])
        ci = concordance_index(given, inferred)
        assert np.isclose(ci, 1.0), f"Perfect concordance failed: {ci:.4f}"

    def test_perfect_discordance():
        given = np.array([1, 2, 3, 4, 5])
        inferred = np.array([5, 4, 3, 2, 1])
        ci = concordance_index(given, inferred)
        assert np.isclose(ci, 0.0), f"Perfect discordance failed: {ci:.4f}"

    def test_all_inferred_tied():
        given = np.array([1, 2, 3, 4])
        inferred = np.array([5, 5, 5, 5])
        ci = concordance_index(given, inferred)
        expected = (0 + 0.5*6)/6  # 6 admissible pairs, all tied
        assert np.isclose(ci, 0.5), f"All tied failed: {ci:.4f}"

    def test_partial_ties():
        given = np.array([1, 2, 3, 4])
        inferred = np.array([1, 1, 2, 2])
        # Admissible pairs: 6
        # Concordant: (0,2), (0,3), (1,2), (1,3) = 4
        # Tied: (0,1), (2,3) = 2
        expected = (4 + 0.5*2)/6
        ci = concordance_index(given, inferred)
        assert np.isclose(ci, expected, atol=1e-4), f"Partial ties failed: {ci:.4f} vs {expected:.4f}"

    def test_complex_case():
        given = np.array([1, 3, 2, 4])
        inferred = np.array([1, 2, 3, 4])
        # Admissible pairs: 6
        # Concordant: 5 pairs (all except (3,2))
        expected = 5/6
        ci = concordance_index(given, inferred)
        assert np.isclose(ci, expected, atol=1e-4), f"Complex case failed: {ci:.4f} vs {expected:.4f}"

    def test_large_scale_validation():
        # Test with 1000 elements (permutation test)
        np.random.seed(42)
        size = 1000
        given = np.arange(size)
        
        # Perfect case
        inferred = given + np.random.uniform(-0.1, 0.1, size)
        ci = concordance_index(given, inferred)
        assert np.isclose(ci, 1.0, atol=1e-4), f"Large perfect failed: {ci:.4f}"

        # Reverse order
        ci = concordance_index(given, np.flip(given))
        assert np.isclose(ci, 0.0, atol=1e-4), f"Large reverse failed: {ci:.4f}"

        # Random permutation (expected ~0.5)
        inferred = np.random.permutation(given)
        ci = concordance_index(given, inferred)
        assert 0.48 < ci < 0.52, f"Random concordance out of range: {ci:.4f}"

    def test_edge_cases():
        # All given tied
        try:
            concordance_index(np.array([2,2,2]), np.array([1,2,3]))
            assert False, "All tied given should raise error"
        except ValueError as e:
            assert "No admissible pairs" in str(e)

        # Minimal valid case
        ci = concordance_index(np.array([1,2]), np.array([1.1, 2.0]))
        assert np.isclose(ci, 1.0), f"Minimal case failed: {ci:.4f}"

    test_perfect_concordance()
    test_perfect_discordance()
    test_all_inferred_tied()
    test_partial_ties()
    test_complex_case()
    test_large_scale_validation()
    test_edge_cases()
    print("All tests passed!")
