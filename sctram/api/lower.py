#!/usr/bin/env python3

import sys
from loguru import logger
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np

# === Inference Modules ===
# (Assumed to be implemented elsewhere in your package.)
from sctram.infer._dpt import DPTInference
from sctram.infer._paga import PAGAInference
from sctram.infer._pca import PCAEmbedding
from sctram.infer._umap import UMAPEmbedding
from sctram.infer._diffmap import DiffmapEmbedding
from sctram.infer._obsm import ObsmEmbedding

# === Evaluation Modules ===
from sctram.evaluate._pseudotime import PseudotimeValuesEvaluation
from sctram.evaluate._adjacency import AdjacencyMatrixEvaluation
from sctram.evaluate._embedding import EmbeddingTrajectoryEvaluation

###############################################################################
# Default Metric Lists
###############################################################################
DEFAULT_PSEUDOTIME_METRICS = [
    "pearson",
    "spearman",
    "kendall",
    "mse",
    "mae",
    "r2",
    "r2_with_spline",
    "concordance_index",
    "dynamic_time_warping",
    "wasserstein_distance",
]

DEFAULT_ADJACENCY_METRICS = [
    "frobenius",
    "L1",
    "accuracy",
    "graph_edit_distance",
    "spectral_distance",
    "jaccard",
    "hamming",
    "precision",
    "recall",
    "f1_score",
    "mantel_correlation",
]

DEFAULT_EMBEDDING_METRICS = [
    "normalized_mean_curvature",
    "distance_correlation",
    "stress",
    "neighborhood_preservation_score",
    "branch_silhouette_score",
    "transition_smoothness",
    "trustworthiness",
    "leaf_node_separation",
    "morans_i",
    "gearys_c",
    "lisa",
    "getis_ord_gi_star",
]

###############################################################################
# Mapping of Method Names to Classes
###############################################################################
PSEUDOTIME_METHODS = {
    "DPTInference": DPTInference,
    # Add other pseudotime inference classes as needed.
}

ADJACENCY_METHODS = {
    "PAGAInference": PAGAInference,
    # Add other adjacency inference classes as needed.
}

EMBEDDING_METHODS = {
    "ObsmEmbedding": ObsmEmbedding,
    "PCAEmbedding": PCAEmbedding,
    "UMAPEmbedding": UMAPEmbedding,
    "DiffmapEmbedding": DiffmapEmbedding,
    # Add any additional embedding methods.
}

###############################################################################
# TrajectoryEvaluationAPI Class
###############################################################################




class TrajectoryEvaluationAPI:
    """
    A comprehensive API for evaluating a user-defined trajectory against
    inferred trajectories from single-cell data. This API supports three
    different evaluation paths:
    
    - Pseudotime evaluation (e.g., using DPTInference)
    - Adjacency/graph evaluation (e.g., using PAGAInference)
    - Embedding evaluation (e.g., using PCA/UMAP/Diffmap/Obsm)
    
    The API is designed to be flexible so that users can provide their own
    inference methods, choose from a set of default metrics, or supply custom
    metric lists.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        trajectory_definition: Any,
        labels: Optional[np.ndarray] = None,
        logger_level: int = str,
    ):
        """
        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object containing your data.
        trajectory_definition : Any
            A representation of the ground truth trajectory. This could be
            an edge list, a graph, or a custom trajectory object.
        labels : np.ndarray, optional
            Cell-level labels (e.g., cluster IDs, experimental conditions, etc.).
        logger_level : int
            Logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self.adata = adata
        self.trajectory = trajectory_definition
        self.labels = labels
        self.results: Dict[str, Any] = {}

        # Setup logger
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level=logger_level)
        self.logger = logger.bind(name="TrajectoryEvaluationAPI")

    def evaluate_pseudotime(
        self,
        method: str = "DPTInference",
        method_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run pseudotime inference and evaluate the inferred trajectory.
        
        Parameters
        ----------
        method : str
            Inference method to use. Default is 'DPTInference'.
        method_params : dict, optional
            Parameters for the chosen pseudotime method.
        metrics : list, optional
            List of metrics to compute. Defaults to a preset list.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of computed pseudotime evaluation metrics.
        """
        self.logger.info("Starting pseudotime evaluation ...")
        method_params = method_params or {}

        if method not in PSEUDOTIME_METHODS:
            raise ValueError(f"Pseudotime method '{method}' not implemented.")

        # Initialize and run inference
        InferenceClass = PSEUDOTIME_METHODS[method]
        self.logger.info("Running pseudotime inference with method '%s'", method)
        inference = InferenceClass(adata=self.adata, labels=self.labels, **method_params)
        inference.calculate()
        inferred_pseudotime = inference.get_result("vector")

        # Run evaluation
        metrics = metrics or DEFAULT_PSEUDOTIME_METRICS
        evaluator = PseudotimeValuesEvaluation(method_params={"metrics": metrics})
        self.logger.info("Evaluating pseudotime metrics ...")
        evaluator.evaluate(
            given_trajectory=self.trajectory,
            inferred_trajectory=inferred_pseudotime,
            labels=self.labels,
        )
        self.results["pseudotime"] = evaluator.get_result()
        self.logger.info("Pseudotime evaluation complete.")
        return self.results["pseudotime"]

    def evaluate_adjacency(
        self,
        method: str = "PAGAInference",
        method_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run adjacency/graph inference and evaluate the inferred graph.
        
        Parameters
        ----------
        method : str
            Inference method to use. Default is 'PAGAInference'.
        method_params : dict, optional
            Parameters for the chosen adjacency method.
        metrics : list, optional
            List of metrics to compute. Defaults to a preset list.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of computed adjacency evaluation metrics.
        """
        self.logger.info("Starting adjacency evaluation ...")
        method_params = method_params or {}

        if method not in ADJACENCY_METHODS:
            raise ValueError(f"Adjacency method '{method}' not implemented.")

        InferenceClass = ADJACENCY_METHODS[method]
        self.logger.info("Running adjacency inference with method '%s'", method)
        inference = InferenceClass(adata=self.adata, labels=self.labels, **method_params)
        inference.calculate()
        inferred_adjacency = inference.get_result("adjacency")
        inferred_labels = inference.get_result("labels")

        metrics = metrics or DEFAULT_ADJACENCY_METRICS
        evaluator = AdjacencyMatrixEvaluation(method_params={"metrics": metrics})
        self.logger.info("Evaluating adjacency metrics ...")
        evaluator.evaluate(
            given_trajectory=self.trajectory,
            inferred_trajectory=inferred_adjacency,
            labels=inferred_labels,
        )
        self.results["adjacency"] = evaluator.get_result()
        self.logger.info("Adjacency evaluation complete.")
        return self.results["adjacency"]

    def evaluate_embedding(
        self,
        method: str = "ObsmEmbedding",
        method_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run an embedding inference and evaluate how well the embedding preserves
        the user-defined trajectory.
        
        Parameters
        ----------
        method : str
            Inference method to use. Default is 'ObsmEmbedding'. Other options
            include 'PCAEmbedding', 'UMAPEmbedding', or 'DiffmapEmbedding'.
        method_params : dict, optional
            Parameters for the chosen embedding method.
        metrics : list, optional
            List of metrics to compute. Defaults to a preset list.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of computed embedding evaluation metrics.
        """
        self.logger.info("Starting embedding evaluation ...")
        method_params = method_params or {}

        if method not in EMBEDDING_METHODS:
            raise ValueError(f"Embedding method '{method}' not implemented.")

        InferenceClass = EMBEDDING_METHODS[method]
        self.logger.info("Running embedding inference with method '%s'", method)
        inference = InferenceClass(adata=self.adata, labels=self.labels, **method_params)
        inference.calculate()

        # Here we assume that each embedding class defines an attribute
        # "output_key" that tells us where the computed embedding is stored.
        embedded_coords = inference.get_result(inference.output_key)

        metrics = metrics or DEFAULT_EMBEDDING_METRICS
        evaluator = EmbeddingTrajectoryEvaluation(method_params={"metrics": metrics})
        self.logger.info("Evaluating embedding metrics ...")
        evaluator.evaluate(
            given_trajectory=self.trajectory,
            inferred_trajectory=embedded_coords,
            labels=self.labels,
        )
        self.results["embedding"] = evaluator.get_result()
        self.logger.info("Embedding evaluation complete.")
        return self.results["embedding"]

    def get_all_results(self) -> Dict[str, Any]:
        """
        Return all stored evaluation results.
        """
        return self.results
