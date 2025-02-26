#!/usr/bin/env python3

import numpy as np
import fastdtw
from sctram.evaluate._metrics.validators import validate_zero_or_positive as _validator
from sctram.evaluate._metrics.utils import prepare_pseudotime


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
    # Measure absolute distances. Normalization focuses on shape rather than magnitude.
    given_pseudotime_array = prepare_pseudotime(given_pseudotime_array, method="minmax")
    inferred_pseudotime_array = prepare_pseudotime(inferred_pseudotime_array, method="minmax")
    
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
