#!/usr/bin/env python3

import sys
from loguru import logger
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np

from sctram.api._class_mapping import *
from sctram.api._defaults_read import get_metrics_by_class as default_metrics
from sctram._constants import labels_key


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
        input_trajectories: Any,
        labels_obs: str,
        root_label: str = None,
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
            Logging level (e.g., DEBUG, INFO, WARNING).
        """
        self.adata = adata
        self.input_trajectories = input_trajectories
        self.labels_obs = labels_obs
        self.root_label = root_label
        
        self.results: Dict[str, Any] = {}

        logger.remove()  # Setup logger
        logger.add(sys.stderr, level=logger_level)
        self.logger = logger.bind(name="TrajectoryEvaluationAPI")

    def evaluate_pseudotime(
        self,
        inference_method: str = "DPTInference",
        evaluate_method: str = "PseudotimeValuesEvaluation",
        inference_params: Optional[Dict[str, Any]] = None,
        evaluate_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        
        self.logger.info("Starting pseudotime evaluation.")
        
        if inference_method not in PSEUDOTIME_METHODS:
            raise ValueError(f"Pseudotime method '{inference_method}' not implemented.")
        InferenceClass = PSEUDOTIME_METHODS[inference_method]
        self.logger.info(f"Running pseudotime inference with method {inference_method!r}")
        
        if evaluate_method not in EVALUATION_METHODS:
            raise ValueError(f"Evaluation method '{evaluate_method}' not implemented.")
        EvaluateClass = EVALUATION_METHODS[evaluate_method]
        self.logger.info(f"Running pseudotime evaluation with method {evaluate_method!r}")

        if self.root_label is None or self.root_label not in self.adata.obs[self.labels_obs].to_numpy():
            raise ValueError("Root label is required for pseudotime based metrics.")

        inference_params = inference_params or dict(
            random_state=42, 
            neighbors_params={"n_neighbors": 90},
            iroot_params=dict(
                label_key = labels_key,
                label = self.root_label,
                method = "min_diffmap",  
                outlier_definition_z = 3 
            )
        )
        inference = InferenceClass(adata=self.adata, labels=self.adata.obs[self.labels_obs], **inference_params)
        inference.calculate()
        inferred_trajectories = inference.get_result("vector")
        
        metrics = metrics or default_metrics(evaluate_class=evaluate_method, infer_class=inference_method)
        if metrics is None or not isinstance(metrics, list) or not len(metrics) > 0:
            raise ValueError
        evaluate_params = evaluate_params or dict(
            prepare_params=dict(
                converter = dict(
                    method = "diffusion_with_damping",
                    handle_disconnected = "assign_max_plus_one",
                    alternative_distance = None,
                    root_label = self.root_label
                )
            ),
            subset_params = None
        )
        evaluation = EvaluateClass(method_params = dict(metrics = metrics), **evaluate_params)
        evaluation.evaluate(
            given_trajectory=self.input_trajectories,
            inferred_trajectory=inferred_trajectories,
            labels=self.adata.obs[self.labels_obs].to_numpy()
        )
        
        self.results["pseudotime"] = evaluation.get_result()
        
        
    # def evaluate_adjacency(
    #     self,
    #     method: str = "PAGAInference",
    #     method_params: Optional[Dict[str, Any]] = None,
    #     metrics: Optional[List[str]] = None,
    # ) -> Dict[str, Any]:
    #     """
    #     Run adjacency/graph inference and evaluate the inferred graph.
        
    #     Parameters
    #     ----------
    #     method : str
    #         Inference method to use. Default is 'PAGAInference'.
    #     method_params : dict, optional
    #         Parameters for the chosen adjacency method.
    #     metrics : list, optional
    #         List of metrics to compute. Defaults to a preset list.
        
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         Dictionary of computed adjacency evaluation metrics.
    #     """
    #     self.logger.info("Starting adjacency evaluation ...")
    #     method_params = method_params or {}

    #     if method not in ADJACENCY_METHODS:
    #         raise ValueError(f"Adjacency method '{method}' not implemented.")

    #     InferenceClass = ADJACENCY_METHODS[method]
    #     self.logger.info("Running adjacency inference with method '%s'", method)
    #     inference = InferenceClass(adata=self.adata, labels=self.labels_obs, **method_params)
    #     inference.calculate()
    #     inferred_adjacency = inference.get_result("adjacency")
    #     inferred_labels = inference.get_result("labels")

    #     metrics = metrics or DEFAULT_ADJACENCY_METRICS
    #     evaluator = AdjacencyMatrixEvaluation(method_params={"metrics": metrics})
    #     self.logger.info("Evaluating adjacency metrics ...")
    #     evaluator.evaluate(
    #         given_trajectory=self.input_trajectories,
    #         inferred_trajectory=inferred_adjacency,
    #         labels=inferred_labels,
    #     )
    #     self.results["adjacency"] = evaluator.get_result()
    #     self.logger.info("Adjacency evaluation complete.")
    #     return self.results["adjacency"]

    # def evaluate_embedding(
    #     self,
    #     method: str = "ObsmEmbedding",
    #     method_params: Optional[Dict[str, Any]] = None,
    #     metrics: Optional[List[str]] = None,
    # ) -> Dict[str, Any]:
    #     """
    #     Run an embedding inference and evaluate how well the embedding preserves
    #     the user-defined trajectory.
        
    #     Parameters
    #     ----------
    #     method : str
    #         Inference method to use. Default is 'ObsmEmbedding'. Other options
    #         include 'PCAEmbedding', 'UMAPEmbedding', or 'DiffmapEmbedding'.
    #     method_params : dict, optional
    #         Parameters for the chosen embedding method.
    #     metrics : list, optional
    #         List of metrics to compute. Defaults to a preset list.
        
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         Dictionary of computed embedding evaluation metrics.
    #     """
    #     self.logger.info("Starting embedding evaluation ...")
    #     method_params = method_params or {}

    #     if method not in EMBEDDING_METHODS:
    #         raise ValueError(f"Embedding method '{method}' not implemented.")

    #     InferenceClass = EMBEDDING_METHODS[method]
    #     self.logger.info("Running embedding inference with method '%s'", method)
    #     inference = InferenceClass(adata=self.adata, labels=self.labels_obs, **method_params)
    #     inference.calculate()

    #     # Here we assume that each embedding class defines an attribute
    #     # "output_key" that tells us where the computed embedding is stored.
    #     embedded_coords = inference.get_result(inference.output_key)

    #     metrics = metrics or DEFAULT_EMBEDDING_METRICS
    #     evaluator = EmbeddingTrajectoryEvaluation(method_params={"metrics": metrics})
    #     self.logger.info("Evaluating embedding metrics ...")
    #     evaluator.evaluate(
    #         given_trajectory=self.input_trajectories,
    #         inferred_trajectory=embedded_coords,
    #         labels=self.labels_obs,
    #     )
    #     self.results["embedding"] = evaluator.get_result()
    #     self.logger.info("Embedding evaluation complete.")
    #     return self.results["embedding"]

    # def get_all_results(self) -> Dict[str, Any]:
    #     """
    #     Return all stored evaluation results.
    #     """
    #     return self.results
