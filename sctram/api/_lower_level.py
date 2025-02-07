#!/usr/bin/env python3

import sys
from loguru import logger
from typing import Any, Dict, List, Optional

import pandas as pd
import anndata as ad

from sctram.api._class_mapping import *
from sctram.api._defaults_read import get_metrics_by_class as default_metrics
from sctram._constants import labels_key


class TrajectoryEvaluationAPI:
    """An API for evaluating a user-defined trajectory against inferred trajectories from single-cell data. 
    
    This API supports three different evaluation paths:
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

    def get_all_results(self) -> Dict[str, Any]:
        df_list = []
        for path, metrics in self.results.items():
            for metric, score in metrics.items():
                df_list.append({"path": path, "metric": metric, "score": score})
        df = pd.DataFrame(df_list)
        return df

    def _get_evaluate_method(self, evaluate_method):
        if evaluate_method not in EVALUATION_METHODS:
            raise ValueError(f"Evaluation method '{evaluate_method}' not implemented.")
        EvaluateClass = EVALUATION_METHODS[evaluate_method]
        self.logger.info(f"Running pseudotime evaluation with method {evaluate_method!r}")
        return EvaluateClass
    
    def _get_inference_method(self, inference_method, methods_constant):
        if inference_method not in methods_constant:
            raise ValueError(f"Pseudotime method '{inference_method}' not implemented.")
        InferenceClass = methods_constant[inference_method]
        self.logger.info(f"Running pseudotime inference with method {inference_method!r}")
        return InferenceClass

    def _get_metrics(self, metrics, inference_method, evaluate_method):
        metrics = metrics or default_metrics(evaluate_class=evaluate_method, infer_class=inference_method)
        if metrics is None or not isinstance(metrics, list) or not len(metrics) > 0:
            raise ValueError
        return metrics

    def evaluate_with_defaults(self):
        self.evaluate_adjacency()
        self.evaluate_pseudotime()
        self.evaluate_embedding()

    def evaluate_pseudotime(
        self,
        inference_method: str = "DPTInference",
        evaluate_method: str = "PseudotimeValuesEvaluation",
        inference_params: Optional[Dict[str, Any]] = None,
        evaluate_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
    ):
        
        self.logger.info("Starting pseudotime evaluation.")
        InferenceClass = self._get_inference_method(inference_method, PSEUDOTIME_INFER_METHODS)
        EvaluateClass = self._get_evaluate_method(evaluate_method)
        metrics = self._get_metrics(metrics, inference_method, evaluate_method)

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
        
    def evaluate_adjacency(
        self,
        inference_method: str = "PAGAInference",
        evaluate_method: str = "AdjacencyMatrixEvaluation",
        inference_params: Optional[Dict[str, Any]] = None,
        evaluate_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
    ):
        
        self.logger.info("Starting adjacency evaluation.") 
        InferenceClass = self._get_inference_method(inference_method, ADJACENCY_INFER_METHODS)
        EvaluateClass = self._get_evaluate_method(evaluate_method)
        metrics = self._get_metrics(metrics, inference_method, evaluate_method)
        
        inference_params = inference_params or dict(
            random_state=42, 
            neighbors_params={"n_neighbors": 90},
        )
        inference = InferenceClass(adata=self.adata, labels=self.adata.obs[self.labels_obs], **inference_params)
        inference.calculate()
        inferred_trajectories = inference.get_result("adjacency")
        inferred_trajectories_labels = inference.get_result("labels")
        
        evaluate_params = evaluate_params or dict(
            subset_params = None
        )
        evaluation = EvaluateClass(method_params = dict(metrics = metrics), **evaluate_params)
        evaluation.evaluate(
            given_trajectory=self.input_trajectories,
            inferred_trajectory=inferred_trajectories,
            labels=inferred_trajectories_labels
        )
        
        self.results["adjacency"] = evaluation.get_result()
    
    def evaluate_embedding(
        self,
        inference_method: str = "ObsmEmbedding",
        evaluate_method: str = "EmbeddingTrajectoryEvaluation",
        inference_params: Optional[Dict[str, Any]] = None,
        evaluate_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
    ):
        
        self.logger.info("Starting embedding evaluation.") 
        InferenceClass = self._get_inference_method(inference_method, EMBEDDING_INFER_METHODS)
        EvaluateClass = self._get_evaluate_method(evaluate_method)
        metrics = self._get_metrics(metrics, inference_method, evaluate_method)
        
        inference_params = inference_params or dict(
            random_state=42, 
            obsm_key="X"
        )
        inference = InferenceClass(adata=self.adata, labels=self.adata.obs[self.labels_obs], **inference_params)
        inference.calculate()  # for obsm it does not do anything.
        inferred_trajectories = inference.get_result("obsm")
                
        evaluate_params = evaluate_params or dict()
        evaluation = EvaluateClass(method_params = dict(metrics = metrics), **evaluate_params)
        evaluation.evaluate(
            given_trajectory=self.input_trajectories,
            inferred_trajectory=inferred_trajectories,
            labels=self.adata.obs[self.labels_obs].to_numpy()
        )
        
        self.results["embedding"] = evaluation.get_result() 
