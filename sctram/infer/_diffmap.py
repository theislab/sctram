#!/usr/bin/env python3

from typing import Any, Dict, Optional, Union

import numpy as np
from anndata import AnnData
from pandas import DataFrame, Series

from sctram._constants import x_diffmap_key
from sctram.infer._base import EmbeddingBase


class DiffmapEmbedding(EmbeddingBase):
    """Diffusion Map embedding method.

    This subclass of InferenceAndEmbeddingBase provides functionality to compute the Diffusion Map
    of the dataset. It does not handle DPT-specific parameters such as root selection.
    """

    def __init__(
        self,
        neighbors_params: Optional[Dict[str, Any]] = None,
        diffmap_params: Optional[Dict[str, Any]] = None,
        # Inherited
        random_state: Optional[int] = None,
        adata: Optional[AnnData] = None,
        embedding: Optional[Union[np.ndarray, DataFrame]] = None,
        labels: Optional[Union[np.ndarray, Series]] = None,
        connectivities: Optional[Union[np.ndarray, DataFrame]] = None,
        distances: Optional[Union[np.ndarray, DataFrame]] = None,
        neighbour_key: Optional[str] = None,
    ):
        """Initializes DiffmapEmbedding.

        Args:
            neighbors_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.neighbors`.
            diffmap_params (Optional[Dict[str, Any]]): Parameters for the specific trajectory method.
            random_state (Optional[int], optional): See `TrajectoryEmbeddingBase.__init__`.
            adata (Optional[AnnData], optional): See `TrajectoryEmbeddingBase.__init__`.
            embedding (Optional[Union[np.ndarray, DataFrame]], optional):See `TrajectoryEmbeddingBase.__init__`.
            labels (Optional[Union[np.ndarray, Series]], optional): See `TrajectoryEmbeddingBase.__init__`.
            connectivities (Optional[Union[np.ndarray, DataFrame]], optional): See `TrajectoryEmbeddingBase.__init__`.
            distances (Optional[Union[np.ndarray, DataFrame]], optional): See `TrajectoryEmbeddingBase.__init__`.
            neighbour_key (Optional[str], optional): See `TrajectoryEmbeddingBase.__init__`.
        """
        super().__init__(random_state, adata, embedding, labels, connectivities, distances, neighbour_key)
        self.diffmap_params = diffmap_params or {}
        self.neighbors_params = neighbors_params or {}

    def _calculate(self):
        """Performs the Diffusion Map calculation."""
        self._needs_neighbors(neighbors_params=self.neighbors_params)
        self._needs_diffmap(diffmap_params=self.diffmap_params)

    def get_result(self, return_mode: str) -> Union[AnnData, Any]:
        """Retrieves the result of the Diffusion Map calculation.

        Args:
            return_mode (str): Decide the returned object.
                - 'anndata': Returns the AnnData object with Diffusion Map results.
                - 'diffmap': Returns the Diffusion Map embedding as a numpy array.

        Raises:
            ValueError: If `return_mode` is invalid.
            RuntimeError: If trajectory inference has not been performed yet.

        Returns:
            Union[AnnData, Any]: The result of the Diffusion Map calculation.
        """
        if not hasattr(self, "adata_prepared"):
            raise RuntimeError("First run `calculate`.")

        if return_mode == "anndata":
            return self.adata_prepared
        elif return_mode == "diffmap":
            if x_diffmap_key not in self.adata_prepared.obsm:
                raise RuntimeError("Diffusion Map has not been computed.")
            return self.adata_prepared.obsm[x_diffmap_key]
        else:
            raise ValueError("Invalid 'return_mode'. Choose either 'anndata' or 'diffmap'.")
