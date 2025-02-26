#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sctram.evaluate._metrics._src.validators import validate_inclusive_between_0_1 as _validator
from sctram.evaluate._metrics._src.utils import prepare_pseudotime



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
    norm_given = prepare_pseudotime(given_pseudotime_array, method="zscore")
    norm_inferred = prepare_pseudotime(inferred_pseudotime_array, method="zscore")

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
