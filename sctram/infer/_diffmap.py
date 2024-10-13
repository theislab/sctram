#!/usr/bin/env python3

from typing import Any, Dict, Optional, Union

import scanpy as sc
from anndata import AnnData

from sctram._constants import x_diffmap_key
from sctram.infer._base import TrajectoryInferenceBase


class DiffmapInference(TrajectoryInferenceBase):
    """Diffusion Map trajectory inference method.

    This subclass of TrajectoryInferenceBase provides functionality to compute the Diffusion Map
    of the dataset. It does not handle DPT-specific parameters such as root selection.
    """

    def __init__(
        self,
        neighbors_params: Optional[Dict[str, Any]] = None,
        diffmap_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ):
        """Initializes the DiffmapInference method with optional parameters for neighbors and Diffusion Map.

        Args:
            neighbors_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.neighbors`.
            diffmap_params (Optional[Dict[str, Any]]): Parameters for `sc.tl.diffmap`.
            random_state (Optional[int]): Random state for reproducibility.
        """
        super().__init__(neighbors_params=neighbors_params, method_params=diffmap_params, random_state=random_state)
        self.diffmap_params = diffmap_params or {}

    def _calculate(self):
        """Performs the Diffusion Map calculation."""
        # Compute Diffusion Map
        if x_diffmap_key not in self.adata_prepared.obsm:
            self.logger.info("Computing Diffusion Map.")
            sc.tl.diffmap(self.adata_prepared, **self.diffmap_params)
            self.logger.debug("Diffusion Map computed successfully.")
        else:
            self.logger.info("Diffusion Map is already calculated.")

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
            raise RuntimeError("First run `infer_trajectory`.")

        if return_mode == "anndata":
            return self.adata_prepared
        elif return_mode == "diffmap":
            if x_diffmap_key not in self.adata_prepared.obsm:
                raise RuntimeError("Diffusion Map has not been computed.")
            return self.adata_prepared.obsm[x_diffmap_key]
        else:
            raise ValueError("Invalid 'return_mode'. Choose either 'anndata' or 'diffmap'.")
