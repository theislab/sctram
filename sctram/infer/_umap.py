#!/usr/bin/env python3

from typing import Any, Dict, Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame, Series

from sctram.infer._base import EmbeddingBase
from sctram.utils._constants import x_umap_key
from sctram.utils._loguru_scanpy_capture import redirect_scanpy_logs_to_loguru


class UMAPEmbedding(EmbeddingBase):
    """UMAP embedding method.

    This subclass of InferenceAndEmbeddingBase provides functionality to compute the UMAP of the dataset.
    """

    def __init__(
        self,
        neighbors_params: Optional[Dict[str, Any]] = None,
        umap_params: Optional[Dict[str, Any]] = None,
        # Inherited
        random_state: Optional[int] = None,
        adata: Optional[AnnData] = None,
        embedding: Optional[Union[np.ndarray, DataFrame]] = None,
        labels: Optional[Union[np.ndarray, Series]] = None,
        connectivities: Optional[Union[np.ndarray, DataFrame]] = None,
        distances: Optional[Union[np.ndarray, DataFrame]] = None,
        neighbour_key: Optional[str] = None,
    ):
        """Initializes UMAPEmbedding.

        Args:
            neighbors_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.neighbors`.
            umap_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.umap`.
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
        self.umap_params = umap_params or {}

    def _calculate(self):
        """Performs the UMAP calculation."""
        self._needs_neighbors(neighbors_params=self.neighbors_params)
        with redirect_scanpy_logs_to_loguru(custom_caller="sc.tl.umap"):
            sc.tl.umap(self.adata_prepared, **self.umap_params)

    def get_result(self, return_mode: str) -> Union[AnnData, Any]:
        """Retrieves the result of the UMAP calculation.

        Args:
            return_mode (str): Decide the returned object.
                - 'anndata': Returns the AnnData object with UMAP results.
                - 'umap': Returns the UMAP embedding as a numpy array.

        Raises:
            ValueError: If `return_mode` is invalid.
            RuntimeError: If trajectory inference has not been performed yet.

        Returns:
            Union[AnnData, Any]: The result of the UMAP calculation.
        """
        if not hasattr(self, "adata_prepared"):
            raise RuntimeError("First run `calculate`.")

        if return_mode == "anndata":
            return self.adata_prepared
        elif return_mode == "umap":
            if x_umap_key not in self.adata_prepared.obsm:
                raise RuntimeError("UMAP has not been computed.")
            return self.adata_prepared.obsm[x_umap_key]
        else:
            raise ValueError("Invalid 'return_mode'. Choose either 'anndata' or 'umap'.")
