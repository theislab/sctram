#!/usr/bin/env python3

import numpy as np
from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1


def precision(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    threshold: float,
    validate_result: bool
) -> float:
    """Calculates the precision for given and inferred adjacency matrices.

    Parameters:
        given_adjacency_matrix (np.ndarray): The true adjacency matrix of the graph.
        inferred_adjacency_matrix (np.ndarray): The inferred adjacency matrix.
        threshold (float): The threshold to binarize the inferred adjacency matrix.
        validate_result (bool): If True, validates that the precision is non-negative.

    Returns:
        float: The calculated precision value.
    """
    return _precision_recall_f1(
        metric="precision",
        given_adjacency_matrix=given_adjacency_matrix,
        inferred_adjacency_matrix=inferred_adjacency_matrix,
        threshold=threshold,
        validate_result=validate_result
    )


def recall(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    threshold: float,
    validate_result: bool
) -> float:
    """
    Calculates the recall for given and inferred adjacency matrices.

    Parameters:
        given_adjacency_matrix (np.ndarray): The true adjacency matrix of the graph.
        inferred_adjacency_matrix (np.ndarray): The inferred adjacency matrix.
        threshold (float): The threshold to binarize the inferred adjacency matrix.
        validate_result (bool): If True, validates that the recall is non-negative.

    Returns:
        float: The calculated recall value.
    """
    return _precision_recall_f1(
        metric="recall",
        given_adjacency_matrix=given_adjacency_matrix,
        inferred_adjacency_matrix=inferred_adjacency_matrix,
        threshold=threshold,
        validate_result=validate_result
    )


def f1_score(
    given_adjacency_matrix: np.ndarray,
    inferred_adjacency_matrix: np.ndarray,
    threshold: float,
    validate_result: bool
) -> float:
    """
    Calculates the F1 Score for given and inferred adjacency matrices.

    Parameters:
        given_adjacency_matrix (np.ndarray): The true adjacency matrix of the graph.
        inferred_adjacency_matrix (np.ndarray): The inferred adjacency matrix.
        threshold (float): The threshold to binarize the inferred adjacency matrix.
        validate_result (bool): If True, validates that the F1 Score is non-negative.

    Returns:
        float: The calculated F1 Score value.
    """
    return _precision_recall_f1(
        metric="f1_score",
        given_adjacency_matrix=given_adjacency_matrix,
        inferred_adjacency_matrix=inferred_adjacency_matrix,
        threshold=threshold,
        validate_result=validate_result
    )


def _precision_recall_f1(given_adjacency_matrix: np.ndarray,
                        inferred_adjacency_matrix: np.ndarray,
                        metric: str,
                        threshold: float,
                        validate_result: bool) -> float:
    """
    Computes a performance metric (Precision, Recall, or F1 Score) for comparing two square adjacency matrices.
    Designed for a mathematics/statistics audience, this function evaluates the quality of inferred graph structures 
    by thresholding the inferred adjacency matrix and comparing it to the true (given) adjacency matrix.

    Parameters:
        given_adjacency_matrix (np.ndarray): The true adjacency matrix of the graph (binary or continuous values).
        inferred_adjacency_matrix (np.ndarray): The inferred adjacency matrix containing continuous scores.
        metric (str): The metric to compute. Must be one of 'precision', 'recall', or 'f1_score'.
        threshold (float): The threshold value used to binarize the inferred adjacency matrix.
        validate_result (bool): If True, validates that the computed metric is zero or positive using the provided validator.

    Returns:
        float: The computed metric value. For precision and recall, this is the proportion of correct edge predictions;
               for F1 Score, this is the harmonic mean of precision and recall.

    Advantages:
        - Provides clear insight into the quality of graph edge predictions by measuring different aspects of performance.
        - Precision, recall, and F1 Score together capture the balance between false positives and false negatives.

    Limitations:
        - Does not consider true negatives, making it less informative in highly imbalanced graphs.
        - Sensitive to the choice of threshold and class imbalance, which may affect metric stability.

    Interpretation:
        - Precision: The fraction of predicted edges that are actually present in the true graph.
        - Recall: The fraction of true edges that were successfully predicted.
        - F1 Score: The harmonic mean of precision and recall, offering a single measure of overall prediction accuracy.
    """
    # Binarize the inferred matrix using the given threshold
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
    
    # Calculate true positives, predicted positives, and actual positives
    true_positive = np.sum((inferred_binary == 1) & (given_adjacency_matrix == 1))
    predicted_positive = np.sum(inferred_binary == 1)
    actual_positive = np.sum(given_adjacency_matrix == 1)
    
    # Compute precision, recall, and F1 score
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / actual_positive if actual_positive > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Select the metric based on the input argument
    if metric == "precision":
        score = precision
    elif metric == "recall":
        score = recall
    elif metric == "f1_score":
        score = f1
    else:
        raise ValueError(f"Invalid metric {metric!r} for precision/recall/F1 calculation.")
    
    # Validate that the score is zero or positive if requested
    if validate_result:
        validate_inclusive_between_0_1(score=score)
    
    return score
