#!/usr/bin/env python3

from typing import Any, Dict, Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame, Series

from sctram._constants import labels_key
from sctram.infer._base import InferenceBase


class PAGAInference(InferenceBase):
    """PAGA (Partition-based Graph Abstraction) trajectory inference method.

    This subclass of InferenceAndEmbeddingBase provides specific functionality
    for trajectory inference using the PAGA method, which is based on partitioning
    a graph of cells to identify and abstract the trajectories in single-cell data.

    Inherits From:
        InferenceAndEmbeddingBase: Provides base functionality for trajectory inference methods.

    Additional Methods:
        _calculate: Performs the PAGA calculation to identify cell trajectories.
        _get_result: Retrieves and returns the PAGA result/results. Please note that "None" and "anndata"
            keys are handled in the parent class.
    """

    def __init__(
        self,
        # Paga settings
        neighbors_params: Optional[Dict[str, Any]] = None,
        paga_params: Optional[Dict[str, Any]] = None,
        # Inherited
        random_state: Optional[int] = None,
        adata: Optional[AnnData] = None,
        embedding: Optional[Union[np.ndarray, DataFrame]] = None,
        labels: Optional[Union[np.ndarray, Series]] = None,
        connectivities: Optional[Union[np.ndarray, DataFrame]] = None,
        distances: Optional[Union[np.ndarray, DataFrame]] = None,
        neighbour_key: Optional[str] = None,
    ):
        """Initializes PAGA method.

        Args:
            neighbors_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.neighbors`.
            paga_params (Optional[Dict[str, Any]]): Parameters for the specific trajectory method.
            random_state (Optional[int], optional): See `TrajectoryEmbeddingBase.__init__`.
            adata (Optional[AnnData], optional): See `TrajectoryEmbeddingBase.__init__`.
            embedding (Optional[Union[np.ndarray, DataFrame]], optional):See `TrajectoryEmbeddingBase.__init__`.
            labels (Optional[Union[np.ndarray, Series]], optional): See `TrajectoryEmbeddingBase.__init__`.
            connectivities (Optional[Union[np.ndarray, DataFrame]], optional): See `TrajectoryEmbeddingBase.__init__`.
            distances (Optional[Union[np.ndarray, DataFrame]], optional): See `TrajectoryEmbeddingBase.__init__`.
            neighbour_key (Optional[str], optional): See `TrajectoryEmbeddingBase.__init__`.
        """
        super().__init__(random_state, adata, embedding, labels, connectivities, distances, neighbour_key)

        self.neighbors_params = neighbors_params or {}
        self.paga_params = paga_params or {}

    def _calculate(self):
        """Performs the PAGA calculation."""
        self._needs_neighbors(neighbors_params=self.neighbors_params)
        sc.tl.paga(self.adata_prepared, groups=labels_key, **self.paga_params)

    def get_result(self, return_mode: str) -> Union[AnnData, np.ndarray]:
        """Retrieves the result of the PAGA trajectory inference.

        Args:
            return_mode (str): Decide the returned object. Either anndata or the result of the calculation. The key
                `anndata` used to get the anndata with calculations. Other keys are calculation specific.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: It needs that the trajectories are calculated already with `infer_trajectory` method,

        Returns:
            Union[AnnData, np.ndarray]: The result of the specific trajectory inference method.
        """
        if not hasattr(self, "adata_prepared"):
            raise RuntimeError("First run `calculate`.")
        elif return_mode == "anndata":
            return self.adata_prepared
        elif return_mode == "adjacency":
            paga_graph = self.adata_prepared.uns["paga"]["connectivities"].toarray()
            return paga_graph
        elif return_mode == "labels":
            return self.adata_prepared.obs[labels_key].cat.categories
        else:
            raise ValueError("Invalid 'return_mode'.")
