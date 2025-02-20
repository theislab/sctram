#!/usr/bin/env python3

import sys
working_directory = "/Users/kemalinecik/git_nosync/sctram"
sys.path.append(working_directory)

import math
import itertools
from typing import Optional
import numpy as np
from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1
from sctram.utils._utils import Utils as U


def ssim_mean(given_matrix: np.ndarray, inferred_matrix: np.ndarray, validate_result: bool, repetition: int = 10000, seed: int = 0) -> float:
    """Calculates the Structural Similarity Index (SSIM) between two images.

    SSIM is used to measure the similarity between two images. It is particularly useful in contexts where the visual similarity
    of the images is important. This function is adapted to compare two adjacency matrices representing graph structures, assuming
    they are used as intensity images. SSIM is a comprehensive measure that evaluates brightness, contrast, and structure similarity.
    If `repetition` > 0, it averages the SSIM over `repetition` random permutations of the rows and columns (or 
    all permutations if there are fewer than `repetition`). This is useful for graph adjacency matrices where node 
    labels might be permuted.

    Parameters:
        given_matrix (np.ndarray): A numpy array representing the first image or graph adjacency matrix.
        inferred_matrix (np.ndarray): A numpy array representing the second image or graph adjacency matrix.
        repetition (int): Number of permutations to average over. If `repetition=0`, computes SSIM without permutation.
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
        - An SSIM value of 1 indicates no difference between the matrices.
        - Values significantly lower than 1 suggest considerable differences.
        - The method is sensitive to changes in luminance, contrast, and structural information.

    """
    from skimage.metrics import structural_similarity as ssim

    # Validate inputs
    if not isinstance(repetition, int) or repetition < 1:
        raise ValueError("`repetition` must be an integer >= 1.")
    if repetition > 1e4:  # this is to while loop ensure large_strategy below finishes quickly.
        raise ValueError("`repetition` must be lower than '1e4'.")
    
    U.validate_adjacency_matrix(given_matrix)
    U.validate_adjacency_matrix(inferred_matrix)
    if given_matrix.shape != inferred_matrix.shape:
        raise ValueError("Matrices must have the same shape.")
    
    n = given_matrix.shape[0]
    data_range = given_matrix.max() - given_matrix.min()
    total_perms = math.factorial(n)
    max_iterations = min(repetition, total_perms)
    values = []
    local_rng = np.random.RandomState(seed)
    
    if repetition == 1:
        score, _ = ssim(given_matrix, inferred_matrix, full=True, data_range=data_range)
    
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
            permuted_given = given_matrix[perm][:, perm]
            permuted_inferred = inferred_matrix[perm][:, perm]
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
