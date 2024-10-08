#!/usr/bin/env python3

from typing import Any, Dict, Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData

from sctram.infer._base import TrajectoryInferenceBase, labels_key


class PAGAInference(TrajectoryInferenceBase):
    """PAGA (Partition-based Graph Abstraction) trajectory inference method.

    This subclass of TrajectoryInferenceBase provides specific functionality
    for trajectory inference using the PAGA method, which is based on partitioning
    a graph of cells to identify and abstract the trajectories in single-cell data.

    Inherits From:
        TrajectoryInferenceBase: Provides base functionality for trajectory inference methods.

    Additional Methods:
        _calculate: Performs the PAGA calculation to identify cell trajectories.
        _get_result: Retrieves and returns the PAGA result/results. Please note that "None" and "anndata"
            keys are handled in the parent class.
    """

    def __init__(
        self,
        neighbors_params: Optional[Dict[str, Any]] = None,
        paga_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ):
        """Initializes the PAGA method with optional parameters.

        Args:
            neighbors_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.neighbors`.
            paga_params (Optional[Dict[str, Any]]): Parameters for `sc.tl.paga`.
            random_state (Optional[int]): Random state for reproducibility.
        """
        super().__init__(neighbors_params=neighbors_params, method_params=paga_params, random_state=random_state)

    def _calculate(self):
        """Performs the PAGA calculation."""
        sc.tl.paga(self.adata_prepared, groups=labels_key, **self.method_params)

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
        if self.adata_prepared is None:
            raise RuntimeError("First run `infer_trajectory`.")
        elif return_mode == "anndata":
            return self.adata_prepared
        elif return_mode == "adjacency":
            paga_graph = self.adata_prepared.uns["paga"]["connectivities"].toarray()
            return paga_graph
        elif return_mode == "labels":
            return self.adata_prepared.obs[labels_key].cat.categories      
        else:
            raise ValueError("Invalid 'return_mode'.")
