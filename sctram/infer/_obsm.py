#!/usr/bin/env python3

from typing import Any, Dict, Optional, Union

import scanpy as sc
import numpy as np
from anndata import AnnData
from pandas import DataFrame, Series

from sctram.infer._base import EmbeddingBase


class ObsmEmbedding(EmbeddingBase):
    """Dummy embedding method. It is created for code consistency.

    This subclass of InferenceAndEmbeddingBase provides no functionality. It is a dummy class to pick the obsm 
    of interest, generally the latent space.
    """
    
    def __init__(self,
            obsm_key: Optional[str] = None,
            # Inherited
            random_state: Optional[int] = None, 
            adata: Optional[AnnData] = None, 
            embedding: Optional[Union[np.ndarray, DataFrame]] = None, 
            labels: Optional[Union[np.ndarray, Series]] = None, 
            connectivities: Optional[Union[np.ndarray, DataFrame]] = None, 
            distances: Optional[Union[np.ndarray, DataFrame]] = None, 
            neighbour_key: Optional[str] = None
    ):
        """Initializes ObsmEmbedding.

        Args:
            obsm_key (Optional[str]): Which obsm key to get. If "X" tha adata.X will be obtained chosen.
            random_state (Optional[int], optional): See `TrajectoryEmbeddingBase.__init__`.
            adata (Optional[AnnData], optional): See `TrajectoryEmbeddingBase.__init__`.
            embedding (Optional[Union[np.ndarray, DataFrame]], optional):See `TrajectoryEmbeddingBase.__init__`.
            labels (Optional[Union[np.ndarray, Series]], optional): See `TrajectoryEmbeddingBase.__init__`.
            connectivities (Optional[Union[np.ndarray, DataFrame]], optional): See `TrajectoryEmbeddingBase.__init__`.
            distances (Optional[Union[np.ndarray, DataFrame]], optional): See `TrajectoryEmbeddingBase.__init__`.
            neighbour_key (Optional[str], optional): See `TrajectoryEmbeddingBase.__init__`.
        """    
        super().__init__(random_state, adata, embedding, labels, connectivities, distances, neighbour_key)
        self.obsm_key = obsm_key or "X"
        
        if self.obsm_key != "X" and self.obsm_key not in self.adata_prepared.obsm.keys():
            raise ValueError(f"Provided 'obsm_key' {self.obsm_key!r} is not found in the anndata.")

    def _calculate(self):
        """Performs nothing. A dummy method."""
        pass

    def get_result(self, return_mode: str) -> Union[np.ndarray]:
        """Retrieves the result.

        Args:
            return_mode (str): Decide the returned object.
                - 'anndata': Returns the AnnData object.
                - 'obsm': Returns the embedding as a numpy array.

        Raises:
            ValueError: If `return_mode` is invalid.

        Returns:
            Union[AnnData, Any]: The result.
        """
        if return_mode == "anndata":
            return self.adata_prepared
        elif return_mode == "obsm" and self.obsm_key != "X":
            return self.adata_prepared.obsm[self.obsm_key]
        elif return_mode == "obsm" and self.obsm_key == "X":
            return self.adata_prepared.X
        else:
            raise ValueError("Invalid 'return_mode'. Choose either 'anndata' or 'obsm'.")
