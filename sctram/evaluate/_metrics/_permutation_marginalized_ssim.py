#!/usr/bin/env python3

import math
import itertools
from typing import Optional
import numpy as np
from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1
from sctram.utils._utils import Utils as U


def permutation_marginalized_ssim(given_adjacency_matrix: np.ndarray, inferred_adjacency_matrix: np.ndarray, validate_result: bool, permutations: int = 10000, seed: int = 0) -> float:
    """Calculates the Permutation-Marginalized Structural Similarity Index (SSIM) between two images.

    SSIM is used to measure the similarity between two images. It is particularly useful in contexts where the visual similarity
    of the images is important. This function is adapted to compare two adjacency matrices representing graph structures, assuming
    they are used as intensity images. SSIM is a comprehensive measure that evaluates brightness, contrast, and structure similarity.
    If `permutations` > 0, it averages the SSIM over `permutations` random permutations of the rows and columns (or 
    all permutations if there are fewer than `permutations`). This is useful for graph adjacency matrices where node 
    labels might be permuted.
    
    Defense of the Approach:
        - Graphs have no inherent node order, so permuting both matrices marginalizes over labeling noise, 
            making the metric agnostic to arbitrary node indexing.
        - Interpretation: We effectively get the expected SSIM over all (or many) label alignments. If the two 
            graphs are truly structurally similar in “many ways,” their adjacency matrices will tend to have 
            relatively high SSIM under numerous permutations. 
        - One of the biggest flaws in naive adjacency matrix comparisons is ignoring the fact that different node 
            labelings can produce wildly different matrix patterns. By permuting the matrices before computing SSIM, 
            you do address labeling invariance.
        - If two graphs share similar global properties (e.g., degree distribution, community structure), their 
            adjacency matrices may exhibit consistent local spatial patterns 
            across permutations, leading to a higher mean SSIM
        - Unlike traditional SSIM, our approach does not assume nodes are ordered meaningfully, making it suitable for graph data where labels are arbitrary.
        - While not a substitute for graph-theoretical metrics, this method provides complementary insights into spatial pattern consistency.
        - This method extends SSIM to adjacency matrices by making it invariant to node permutations, treating graphs as unordered spatial structures.
        - When It Makes Sense
            - Use Case 1: Comparing graphs where node order is irrelevant, and you care about statistical consistency of spatial patterns (e.g., random graphs with shared properties).
            - Use Case 2: Validating generative models (e.g., GANs) where the goal is to produce adjacency matrices that "look similar" to real ones under arbitrary permutations.

    Parameters:
        given_matrix (np.ndarray): A numpy array representing the first image or graph adjacency matrix.
        inferred_matrix (np.ndarray): A numpy array representing the second image or graph adjacency matrix.
        permutations (int): Number of permutations to average over. If `permutations=0`, computes SSIM without permutation.
        validate_result (bool): A bool deciding whether or not to validate the score.
        seed (int, optional): Seed for deterministic permutations. If provided, ensures reproducibility.

    Returns:
        float: The SSIM index, where 1 indicates perfect similarity and values closer to 0 indicate lesser similarity.

    Raises:
        ValueError: If matrices are not square or have mismatched shapes.

    Advantages:
        - Considers perceptual phenomena, making it highly relevant for visual similarity assessments.
        - Provides a more intuitive measure of similarity than pixel-based differences.

    Limitations:
        - Sensitive to image scaling, shifts, and other transformations.
        - Assumes the matrices represent intensity images, which may not always be appropriate for graph adjacency matrices.

    Interpretation:
        - The average does not directly tell you if there exists some permutation that yields almost perfect 
            matching. It's a measure of overall alignment across many permutations, not the best alignment.
        - An SSIM value of 1 indicates no difference between the matrices.
        - Values significantly lower than 1 suggest considerable differences.
        - The method originally is sensitive to changes in luminance, contrast, and structural information.

    """
    from skimage.metrics import structural_similarity as ssim

    # Validate inputs
    if not isinstance(permutations, int) or permutations < 1:
        raise ValueError("`permutations` must be an integer >= 1.")
    if permutations > 1e4:  # this is to while loop ensure large_strategy below finishes quickly.
        raise ValueError("`permutations` must be lower than '1e4'.")
    
    U.validate_adjacency_matrix(given_adjacency_matrix)
    U.validate_adjacency_matrix(inferred_adjacency_matrix)
    if given_adjacency_matrix.shape != inferred_adjacency_matrix.shape:
        raise ValueError("Matrices must have the same shape.")
    
    n = given_adjacency_matrix.shape[0]
    data_range = given_adjacency_matrix.max() - given_adjacency_matrix.min()
    total_perms = math.factorial(n)
    max_iterations = min(permutations, total_perms)
    values = []
    local_rng = np.random.RandomState(seed)
    
    if permutations == 1:
        score, _ = ssim(given_adjacency_matrix, inferred_adjacency_matrix, full=True, data_range=data_range)
    
    else:
        # Small n (Total permutations ≤ 362,880): Generate all permutations using itertools.permutations, 
        # shuffle them, and select the required number. This avoids duplicate checks and is efficient for small n.
        strategy_threshold = math.factorial(9)
        if total_perms <= strategy_threshold:
            all_perms = list(itertools.permutations(range(n)))
            local_rng.shuffle(all_perms)  # Use local RNG's shuffle
            selected_perms = all_perms[:max_iterations]
            selected_perms = [np.array(p) for p in selected_perms]
        
        # Large n (Total permutations > 362,880): Generate permutations randomly, track uniqueness using a set, 
        # and avoid duplicates. This handles cases where generating all permutations is infeasible.
        else:
            selected_perms = []
            used_perms = set()
            while len(selected_perms) < max_iterations:
                perm = local_rng.permutation(n)  # Use local RNG's permutation
                perm_tuple = tuple(perm)
                if perm_tuple not in used_perms:
                    used_perms.add(perm_tuple)
                    selected_perms.append(perm)
        
        # Compute SSIM for each permutation
        for perm in selected_perms:
            permuted_given = given_adjacency_matrix[perm][:, perm]
            permuted_inferred = inferred_adjacency_matrix[perm][:, perm]
            similarity, _ = ssim(
                permuted_given, permuted_inferred,
                full=True, data_range=data_range
            )
            values.append(similarity)
        
        score = np.mean(values)
        # print(values)
        print(score)
    
    if validate_result:
        validate_inclusive_between_0_1(score=score)
    
    return score
