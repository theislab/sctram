#!/usr/bin/env python3

from typing import Any, Dict, Optional, Union

import scanpy as sc
import numpy as np
from anndata import AnnData
from pandas import DataFrame, Series

from sctram._constants import x_pca_key
from sctram.infer._base import EmbeddingBase


class PCAEmbedding(EmbeddingBase):
    """PCA embedding method.

    This subclass of InferenceAndEmbeddingBase provides functionality to compute the PCA of the dataset. 
    """
    
    def __init__(self,
            pca_params: Optional[Dict[str, Any]] = None,
            # Inherited
            random_state: Optional[int] = None, 
            adata: Optional[AnnData] = None, 
            embedding: Optional[Union[np.ndarray, DataFrame]] = None, 
            labels: Optional[Union[np.ndarray, Series]] = None, 
            connectivities: Optional[Union[np.ndarray, DataFrame]] = None, 
            distances: Optional[Union[np.ndarray, DataFrame]] = None, 
            neighbour_key: Optional[str] = None
    ):
        """Initializes PCAEmbedding.

        Args:
            pca_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.pca`.
            random_state (Optional[int], optional): See `TrajectoryEmbeddingBase.__init__`.
            adata (Optional[AnnData], optional): See `TrajectoryEmbeddingBase.__init__`.
            embedding (Optional[Union[np.ndarray, DataFrame]], optional):See `TrajectoryEmbeddingBase.__init__`.
            labels (Optional[Union[np.ndarray, Series]], optional): See `TrajectoryEmbeddingBase.__init__`.
            connectivities (Optional[Union[np.ndarray, DataFrame]], optional): See `TrajectoryEmbeddingBase.__init__`.
            distances (Optional[Union[np.ndarray, DataFrame]], optional): See `TrajectoryEmbeddingBase.__init__`.
            neighbour_key (Optional[str], optional): See `TrajectoryEmbeddingBase.__init__`.
        """    
        super().__init__(random_state, adata, embedding, labels, connectivities, distances, neighbour_key)
        self.pca_params = pca_params or {}

    def _calculate(self):
        """Performs the PCA calculation."""
        sc.pp.pca(self.adata_prepared, **self.pca_params)

    def get_result(self, return_mode: str) -> Union[AnnData, Any]:
        """Retrieves the result of the PCA calculation.

        Args:
            return_mode (str): Decide the returned object.
                - 'anndata': Returns the AnnData object with PCA results.
                - 'pca': Returns the PCA embedding as a numpy array.

        Raises:
            ValueError: If `return_mode` is invalid.
            RuntimeError: If trajectory inference has not been performed yet.

        Returns:
            Union[AnnData, Any]: The result of the PCA calculation.
        """
        if not hasattr(self, "adata_prepared"):
            raise RuntimeError("First run `calculate`.")

        if return_mode == "anndata":
            return self.adata_prepared
        elif return_mode == "pca":
            if x_pca_key not in self.adata_prepared.obsm:
                raise RuntimeError("PCA has not been computed.")
            return self.adata_prepared.obsm[x_pca_key]
        else:
            raise ValueError("Invalid 'return_mode'. Choose either 'anndata' or 'pca'.")
