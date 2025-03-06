#!/usr/bin/env python3

import sys
from typing import Any, Dict, List, Optional

import gc
import anndata as ad
import pandas as pd
from loguru import logger

from sctram.api._class_mapping import *
from sctram.api._defaults_read import get_metrics_by_class as default_metrics
from sctram.input import InputTrajectories, InputTrajectory
from sctram.utils._constants import labels_key, neighbors_key, key_trajectories


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
        input_trajectory: InputTrajectory,
        labels_obs: str,
        root_label: str = None,
        logger_level: str = "INFO",
    ):
        self.logger_level = logger_level
        self.set_logger()

        self.adata = adata
        self.input_trajectory = input_trajectory
        self.labels_obs = labels_obs
        self.root_label = root_label  # TODO: depracete this and use only for pseudotime evaluation method.

        self.results: Dict[str, Any] = {}

    def set_logger(self):
        logger.remove()  # Setup logger
        logger.add(sys.stderr, level=self.logger_level)
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
        # Main sctram running function, very simplified version api run.
        # TODO: write advanced version of this.
        self.evaluate_embedding()
        adata = self.evaluate_adjacency(_return_inference_anndata=True)
        self.evaluate_pseudotime(_given_adata=adata)

    def evaluate_pseudotime(
        self,
        inference_method: str = "DPTInference",
        evaluate_method: str = "PseudotimeValuesEvaluation",
        inference_params: Optional[Dict[str, Any]] = None,
        evaluate_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        _given_adata: Optional[ad.AnnData] = None,
        _return_inference_anndata=False,
    ):

        self.logger.info("Starting pseudotime evaluation.")
        InferenceClass = self._get_inference_method(inference_method, PSEUDOTIME_INFER_METHODS)
        EvaluateClass = self._get_evaluate_method(evaluate_method)
        metrics = self._get_metrics(metrics, inference_method, evaluate_method)

        # Subset anndata with only available nodes.
        if _given_adata is None:
            _adata = self.adata[self.adata.obs[self.labels_obs].isin(self.input_trajectory.nodes())]
        else:
            _adata = _given_adata

        if self.root_label is None or self.root_label not in _adata.obs[self.labels_obs].to_numpy():
            root_label = self.input_trajectory.get_unique_root()
            self.logger.info("Root label is obtained from the InputTrajectory object.")
        else:
            root_label = self.root_label

        inference_params = inference_params or dict(
            random_state=42,
            neighbors_params={"n_neighbors": 50},
            iroot_params=dict(label_key=labels_key, label=root_label, method="centroid", outlier_definition_z=3),
        )

        if _given_adata is None:
            inference = InferenceClass(adata=_adata, labels=_adata.obs[self.labels_obs], **inference_params)
        else:
            inference = InferenceClass(
                adata=_adata, labels=_adata.obs[self.labels_obs], neighbour_key=neighbors_key, **inference_params
            )

        inference.calculate()
        inferred_trajectories = inference.get_result("vector")

        evaluate_params = evaluate_params or dict(
            prepare_params=dict(
                converter=dict(
                    method="diffusion_with_damping",
                    handle_disconnected="assign_max_plus_one",
                    alternative_distance=None,
                    root_label=root_label,
                )
            ),
            subset_params=None,
        )
        evaluation = EvaluateClass(method_params=dict(metrics=metrics), **evaluate_params)
        evaluation.evaluate(
            given_trajectory=self.input_trajectory,
            inferred_trajectory=inferred_trajectories,
            labels=_adata.obs[self.labels_obs].to_numpy(),
        )

        self.results["pseudotime"] = evaluation.get_result()

        if _return_inference_anndata:
            return inference.get_result("anndata")

    def evaluate_adjacency(
        self,
        inference_method: str = "PAGAInference",
        evaluate_method: str = "AdjacencyMatrixEvaluation",
        inference_params: Optional[Dict[str, Any]] = None,
        evaluate_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        _given_adata: Optional[ad.AnnData] = None,
        _return_inference_anndata=False,
    ):

        self.logger.info("Starting adjacency evaluation.")
        InferenceClass = self._get_inference_method(inference_method, ADJACENCY_INFER_METHODS)
        EvaluateClass = self._get_evaluate_method(evaluate_method)
        metrics = self._get_metrics(metrics, inference_method, evaluate_method)

        inference_params = inference_params or dict(
            random_state=42,
            neighbors_params={"n_neighbors": 50},
        )
        # Subset anndata with only available nodes.
        if _given_adata is None:
            _adata = self.adata[self.adata.obs[self.labels_obs].isin(self.input_trajectory.nodes())]
        else:
            _adata = _given_adata

        if _given_adata is None:
            inference = InferenceClass(adata=_adata, labels=_adata.obs[self.labels_obs], **inference_params)
        else:
            inference = InferenceClass(
                adata=_adata, labels=_adata.obs[self.labels_obs], neighbour_key=neighbors_key, **inference_params
            )

        inference.calculate()
        inferred_trajectories = inference.get_result("adjacency")
        inferred_trajectories_labels = inference.get_result("labels")

        evaluate_params = evaluate_params or dict(subset_params=None)
        evaluation = EvaluateClass(method_params=dict(metrics=metrics), **evaluate_params)
        evaluation.evaluate(
            given_trajectory=self.input_trajectory,
            inferred_trajectory=inferred_trajectories,
            labels=inferred_trajectories_labels,
        )

        self.results["adjacency"] = evaluation.get_result()

        if _return_inference_anndata:
            return inference.get_result("anndata")

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

        inference_params = inference_params or dict(random_state=42, obsm_key="X")
        inference = InferenceClass(adata=self.adata, labels=self.adata.obs[self.labels_obs], **inference_params)
        inference.calculate()  # for obsm it does not do anything.
        inferred_trajectories_anndata = inference.get_result("anndata")
        inference._needs_neighbors(neighbors_params={"n_neighbors": 50})

        evaluate_params = evaluate_params or dict()
        evaluation = EvaluateClass(method_params=dict(metrics=metrics), **evaluate_params)
        evaluation.evaluate(
            given_trajectory=self.input_trajectory,
            inferred_trajectory=inferred_trajectories_anndata,
            labels=self.adata.obs[self.labels_obs].to_numpy(),
        )

        self.results["embedding"] = evaluation.get_result()


class WithBootstrappedTrajectoryAPI:

    def __init__(
        self, 
        input_trajectory: InputTrajectory, 
        decomposition_method: str,
        lower_level_api_kwargs: dict, 
        bootstrap_kwargs: dict = dict(), 
        logger_level: str = "INFO"
    ):
        self.logger_level = logger_level
        self.set_logger()

        if "logger_level" in lower_level_api_kwargs:
            raise ValueError
        if "input_trajectory" in lower_level_api_kwargs:
            raise ValueError
        
        self.lower_level_api_kwargs = lower_level_api_kwargs
        self.input_trajectory = input_trajectory
        self.decomposition_method = decomposition_method
        
        if decomposition_method == "method_1":
            self.input_trajectories_subgraphs = self.input_trajectory.decompose_trajectory_method_1(**bootstrap_kwargs)
        elif decomposition_method == "method_2":
            self.input_trajectories_subgraphs = self.input_trajectory.decompose_trajectory_method_2(**bootstrap_kwargs)
        else:
            raise ValueError
            
        self.results: Dict[str, Any] = {}

    def evaluate_with_defaults(self):
        for it in self.input_trajectories_subgraphs:
            api = TrajectoryEvaluationAPI(logger_level=self.logger_level, input_trajectory=it, **self.lower_level_api_kwargs)
            api.evaluate_with_defaults()
            df = api.get_all_results()
            self.results[it.graph[key_trajectories]] = dict(
                df = df.copy(),
                subgraph = it.copy()
            )
            del api
            gc.collect()
    
    def set_logger(self):
        logger.remove()
        logger.add(sys.stderr, level=self.logger_level)
        self.logger = logger.bind(name="TrajectoryEvaluationWithBootstrappedTrajectoryAPI")
