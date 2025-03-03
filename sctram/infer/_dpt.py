#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame, Series
from scipy.stats import zscore

from sctram.infer._base import InferenceBase
from sctram.utils._constants import iroot_key, labels_key, x_diffmap_key
from sctram.utils._loguru_scanpy_capture import redirect_scanpy_logs_to_loguru


class DPTInference(InferenceBase):
    """DPT (Diffusion Pseudotime) trajectory inference method with customizable root setting.

    This subclass of InferenceAndEmbeddingBase provides enhanced functionality for trajectory inference
    using the DPT method. It allows customization of the Diffusion Map calculation and provides flexible
    methods to set the root of the trajectory.

    Inherits From:
        InferenceAndEmbeddingBase: Provides base functionality for trajectory inference methods.

    Additional Features:
        - Customizable Diffusion Map calculation parameters.
        - Flexible root setting through various methods:
            - Based on Diffusion Map components.
            - Using a custom binary array.
            - Selecting a root based on cell labels.
    """

    def __init__(
        self,
        # Paga settings
        neighbors_params: Optional[Dict[str, Any]] = None,
        diffmap_params: Optional[Dict[str, Any]] = None,
        dpt_params: Optional[Dict[str, Any]] = None,
        iroot_params: Optional[Union[int, np.ndarray, dict]] = None,
        # Inherited
        random_state: Optional[int] = None,
        adata: Optional[AnnData] = None,
        embedding: Optional[Union[np.ndarray, DataFrame]] = None,
        labels: Optional[Union[np.ndarray, Series]] = None,
        connectivities: Optional[Union[np.ndarray, DataFrame]] = None,
        distances: Optional[Union[np.ndarray, DataFrame]] = None,
        neighbour_key: Optional[str] = None,
    ):
        """Initializes the DPT method with optional parameters for neighbors, Diffusion Map, DPT, and root selection.

        Args:
            neighbors_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.neighbors`.
            diffmap_params (Optional[Dict[str, Any]]): Parameters for `sc.tl.diffmap`.
            dpt_params (Optional[Dict[str, Any]]): Parameters for `sc.tl.dpt`.
            iroot_params (Optional[Union[int, np.ndarray, dict]]): Specification for the root of the trajectory.
                - If `int`, it represents the index of the root cell.
                - If `np.ndarray`, it should be a binary array with exactly one `1` indicating the root cell.
                - If `dict`, it represents a cell label from which the root will be selected. Dictionary must contain:
                    - `label` (str): The specific label to select the root from.
                    - `method` (str): The method to select the root within the specified label. Supported methods:
                        - `'min_diffmap'`: Select the cell with the minimum value in a specified DiffMap component.
                        - `'centroid'`: Select the cell closest to the centroid of the cluster.
                        - `'density'`: Select the cell in the cluster with the highest density.
                        - `'random'`: Select a random cell from the cluster.
                    - `outlier_definition_z` (float, optional): Decide whether or not ignore outliers in calculations.
                        If `None`, then all datapoints are taken into consideration.
                    - Additional method-specific parameters:
                        - For `'min_diffmap'`:
                            - `component` (int, optional): The Diffusion Map component to use. Defaults to 0.
                        - For `centroid`:
                            - `centroid_embedding` (str): 'X' or a key from `adata.obsm` to be used to calculate
                                the centroids. Defaults to 'X'.
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
        self.diffmap_params = diffmap_params or {}
        self.dpt_params = dpt_params or {}
        self.iroot_params = iroot_params  # Specification for the root

    def _calculate(self):
        """Performs the Diffusion Map and DPT calculations, including root setting."""
        # Compute Diffusion Map
        self._needs_neighbors(neighbors_params=self.neighbors_params)
        self._needs_diffmap(diffmap_params=self.diffmap_params)

        # Determine and set the root
        if self.iroot_params is not None:
            if isinstance(self.iroot_params, int):
                self.logger.debug(f"Setting root using provided integer index: {self.iroot_params}.")
                self.set_root_custom_index(self.iroot_params)
            elif isinstance(self.iroot_params, np.ndarray):
                self.logger.debug("Setting root using provided custom binary array.")
                self.set_root_custom_array(custom_array=self.iroot_params)
            elif isinstance(self.iroot_params, dict):
                self.logger.debug("Setting root using provided label-based specification.")
                self.set_root_from_label_dict(label_dict=self.iroot_params)
            else:
                raise ValueError(
                    f"Invalid type for 'iroot' {type(self.iroot_params)!r}. Must be int, np.ndarray, or dict."
                )
        else:
            # Default root: cell with minimum value in the 4th Diffusion Map component
            self.logger.info("No root specification provided. Setting default root based on Diffusion Map.")
            self.set_root_from_diffmap()

        # Retrieve the root index
        if iroot_key not in self.adata_prepared.uns:
            raise RuntimeError("Root index ('iroot') has not been set.")

        # Perform DPT with the specified root in `iroot`
        self.logger.info("Performing DPT trajectory calculation.")
        with redirect_scanpy_logs_to_loguru(custom_caller="sc.tl.dpt"):
            sc.tl.dpt(self.adata_prepared, **self.dpt_params)
        self.logger.debug("DPT calculation completed successfully.")

    def get_result(self, return_mode: str) -> Union[AnnData, np.ndarray]:
        """Retrieves the result of the DPT trajectory inference.

        Args:
            return_mode (str): Decide the returned object. Either 'anndata' or 'vector'.
                - 'anndata': Returns the AnnData object with DPT results.
                - 'vector': Returns the pseudotime as a numpy array.

        Raises:
            ValueError: If `return_mode` is invalid.
            RuntimeError: If trajectory inference has not been performed yet.

        Returns:
            Union[AnnData, np.ndarray]: The result of the DPT trajectory inference.
        """
        if self.adata_prepared is None:
            raise RuntimeError("First run `calculate`.")

        if return_mode == "anndata":
            return self.adata_prepared
        elif return_mode == "vector":
            pseudotime = self.adata_prepared.obs["dpt_pseudotime"].values
            return pseudotime
        else:
            raise ValueError("Invalid 'return_mode'. Choose either 'anndata' or 'vector'.")

    def set_root_custom_index(self, root_ix: int):
        """Sets the root using a specific cell index.

        Args:
            root_ix (int): The index of the cell to set as the root.

        Raises:
            ValueError: If the provided index is out of bounds.
        """
        if not (0 <= root_ix < self.adata_prepared.n_obs):
            raise ValueError(
                f"Root index {root_ix} is out of bounds for the dataset with {self.adata_prepared.n_obs} cells."
            )
        self.adata_prepared.uns[iroot_key] = root_ix
        self.logger.info(f"Root set to cell index {root_ix} based on provided index.")

    def set_root_from_diffmap(self, component: int = 0):
        """Sets the root of the trajectory based on the specified Diffusion Map component.

        The root is set to the cell with the minimum value in the specified Diffusion Map component.

        Args:
            component (int): The Diffusion Map component to use for root selection.
                Defaults to 0 (the first component).

        Raises:
            ValueError: If the specified component is out of bounds.
            RuntimeError: If `diffmap_key`, generally `X_diffmap` is not found in the anndata object.
                Happens when diffusion map has not been computed by `sc.tl.diffmap`.
        """
        if x_diffmap_key not in self.adata_prepared.obsm:
            raise RuntimeError("Diffusion Map has not been computed.")

        if component < 0 or component >= self.adata_prepared.obsm[x_diffmap_key].shape[1]:
            raise ValueError(f"Component index {component} is out of bounds for Diffusion Map.")

        root_ix = np.argmin(self.adata_prepared.obsm[x_diffmap_key][:, component])
        self.adata_prepared.uns[iroot_key] = root_ix
        self.logger.info(f"Root set to cell index {root_ix} based on Diffusion Map component {component}.")

    def set_root_custom_array(self, custom_array: np.ndarray):
        """Sets the root of the trajectory using a custom binary array.

        The array must contain exactly one `1` indicating the root cell, and all other entries must be `0`.

        Args:
            custom_array (np.ndarray): A binary array with exactly one `1`.

        Raises:
            ValueError: If the array does not contain exactly one `1`, or its length does not match the number of cells.
            TypeError: If the array is not numpy array.
        """
        if not isinstance(custom_array, np.ndarray):
            raise TypeError("Custom root specification must be a numpy array.")

        if custom_array.ndim != 1:
            raise ValueError("Custom root array must be one-dimensional.")

        if len(custom_array) != self.adata_prepared.n_obs:
            raise ValueError("Custom root array length must match the number of cells.")

        ones = np.sum(custom_array == 1)
        if ones != 1:
            raise ValueError("Custom root array must contain exactly one '1'.")

        root_ix = np.argmax(custom_array)
        self.adata_prepared.uns[iroot_key] = root_ix
        self.logger.info(f"Root set to cell index {root_ix} based on custom binary array.")

    def set_root_from_label_dict(self, label_dict: Dict[str, Any]):
        """Determines the root index based on cell labels.

        This method selects a root cell within the specified cluster label. The selection can be based on
        various strategies such as the minimum or maximum value in a specified Diffusion Map component,
        the centroid of the cluster, the highest density region, or a random selection.

        Args:
            label_dict (Dict[str, Any]): Dictionary containing label-based root specification.
                See `init` method for specifications.
        """
        root_ix, info = self._set_root_from_label_dict(label_dict=label_dict)
        self.adata_prepared.uns[iroot_key] = root_ix
        self.logger.info(info)

    def _set_root_from_label_dict(self, label_dict: Dict[str, Any], component_default: int = 0) -> Tuple[int, str]:
        """Method providing the root. See `set_root_from_label_dict` for details."""  # noqa
        required_keys = {"label", "method", "outlier_definition_z"}
        if not required_keys.issubset(label_dict.keys()):
            missing = required_keys - label_dict.keys()
            raise ValueError(f"Missing keys in label specification dictionary: {missing}")

        label = label_dict["label"]
        method = label_dict["method"]
        outlier_definition_z = label_dict["outlier_definition_z"]
        optional_params_label_dict = {k: v for k, v in label_dict.items() if k not in required_keys}
        invalid_method_error = (
            "Invalid method for setting root. Choose from 'min_diffmap', 'centroid', 'density', or 'random'."
        )

        if labels_key not in self.adata_prepared.obs:
            raise ValueError(f"Label key {labels_key!r} not found in `adata.obs`.")

        if x_diffmap_key not in self.adata_prepared.obsm:
            raise RuntimeError("Diffusion Map has not been computed.")

        if label not in self.adata_prepared.obs[labels_key].unique():
            raise ValueError(f"Label {label!r} not found in `adata.obs[{labels_key!r}]`.")

        cluster_indices = np.where(self.adata_prepared.obs[labels_key] == label)[0]
        if len(cluster_indices) == 0:
            raise ValueError(f"No cells found for label {label!r}.")

        master_indices = cluster_indices.copy()  # Initialize master_indices

        def _get_comp():
            _comp = optional_params_label_dict.get("component", component_default)
            if _comp < 0 or _comp >= self.adata_prepared.obsm[x_diffmap_key].shape[1]:
                raise ValueError(f"Component index {_comp} is out of bounds for Diffusion Map.")
            return _comp

        def _get_centroid_emb(subset):
            _centroid_emb = optional_params_label_dict.get("centroid_embedding", "X")
            if not (_centroid_emb == "X" or _centroid_emb in self.adata_prepared.obsm.keys()):
                raise ValueError(f"Centroid embedding key {_centroid_emb!r} is not found in `adata.obsm`.")
            _embedding = self.adata_prepared.X if _centroid_emb == "X" else self.adata_prepared.obsm[_centroid_emb]
            return _embedding[subset]

        if outlier_definition_z is not None:
            if method in ["min_diffmap", "density"]:
                # Extract the specified Diffusion Map component for the cluster
                component = _get_comp()
                embedding = self.adata_prepared.obsm[x_diffmap_key][cluster_indices, component]

                # Compute z-scores for the embedding
                if np.std(embedding) == 0:
                    self.logger.warning(
                        f"Standard deviation of the DiffMap component {component} is zero. "
                        "Outlier removal based on z-score is skipped."
                    )
                    master_indices = cluster_indices
                else:
                    z_scores = zscore(embedding)
                    mask = np.abs(z_scores) <= outlier_definition_z
                    refined_indices = cluster_indices[mask]
                    if len(refined_indices) == 0:
                        self.logger.warning(
                            f"All cells in label {label!r} were identified as outliers based on z-score threshold "
                            f"{outlier_definition_z}. Proceeding with all cluster cells."
                        )
                        master_indices = cluster_indices
                    else:
                        master_indices = refined_indices
                        self.logger.info(
                            f"Outlier removal applied. {len(master_indices)} out of {len(cluster_indices)} cells remain "
                            f"after filtering based on z-score <= {outlier_definition_z}."
                        )

            elif method == "centroid":
                # Extract the specified embedding for the cluster
                embedding = _get_centroid_emb(subset=cluster_indices)  # shape: (n_cells, n_features)

                # Compute z-scores for each dimension
                if embedding.ndim != 2:
                    raise ValueError("Centroid embedding must be two-dimensional.")

                z_scores = zscore(embedding, axis=0)
                # Handle the case where a feature has zero variance
                if np.isnan(z_scores).any():
                    self.logger.warning(
                        "One or more dimensions in the centroid embedding have zero variance. "
                        "Outlier removal based on z-score is partially skipped."
                    )
                    z_scores = np.nan_to_num(z_scores, nan=0.0)

                # Create a mask where all z-scores are within the threshold
                mask = np.all(np.abs(z_scores) <= outlier_definition_z, axis=1)
                refined_indices = cluster_indices[mask]
                if len(refined_indices) == 0:
                    self.logger.warning(
                        f"All cells in label {label!r} were identified as outliers based on z-score threshold "
                        f"{outlier_definition_z}. Proceeding with all cluster cells."
                    )
                    master_indices = cluster_indices
                else:
                    master_indices = refined_indices
                    self.logger.info(
                        f"Outlier removal applied. {len(master_indices)} out of {len(cluster_indices)} cells remain "
                        f"after filtering based on z-score <= {outlier_definition_z}."
                    )
            else:
                raise ValueError(invalid_method_error)
        else:
            master_indices = cluster_indices
            self.logger.info(f"No outlier removal applied. Using all {len(master_indices)} cells in label {label!r}.")

        if method == "min_diffmap":
            component = _get_comp()
            cluster_diffmap = self.adata_prepared.obsm[x_diffmap_key][master_indices, component]
            root_sub_ix = np.argmin(cluster_diffmap)
            root_ix = master_indices[root_sub_ix]
            info = (
                f"Root set to cell index {root_ix} based on minimum in "
                f"DiffMap component {component} within label {label!r}."
            )
            return root_ix, info

        elif method == "centroid":
            embedding = _get_centroid_emb(subset=master_indices)  # shape: (n_master_cells, n_features)
            centroid = np.mean(embedding, axis=0)
            distances = np.linalg.norm(embedding - centroid, axis=1)
            root_sub_ix = np.argmin(distances)
            root_ix = master_indices[root_sub_ix]
            info = f"Root set to cell index {root_ix} closest to centroid within label {label!r}."
            return root_ix, info

        elif method == "density":
            if "distances" not in self.adata_prepared.obsp:
                raise RuntimeError(
                    "Distances matrix has not been computed. Cannot compute density. Call scanpy `neighbor` method."
                )
            distances_matrix = self.adata_prepared.obsp["distances"][np.ix_(master_indices, master_indices)].toarray()

            # Compute local density as the number of neighbors within a certain radius
            densities_ = self._adjust_radius_for_density(distances_matrix, max_iterations=100)
            densities = densities_.sum(axis=0)
            self.adata_prepared.uns["_sctram_densities"] = densities_

            if np.all(densities == 0):
                RuntimeError("Unexpected behavior.")

            root_sub_ix = np.argmax(densities)
            root_ix = master_indices[root_sub_ix]
            info = f"Root set to cell index {root_ix} with highest density within label {label!r}."
            return root_ix, info

        elif method == "random":
            rng = np.random.default_rng(self.random_state)
            root_ix = rng.choice(master_indices)
            info = f"Root set to randomly selected cell index {root_ix} within label {label!r}."
            return root_ix, info

        else:
            raise ValueError(invalid_method_error)

    def _adjust_radius_for_density(
        self,
        distances_matrix: np.ndarray,
        max_iterations: int = 1000,
        error_margin: float = 1,
        initial_step_ratio: float = 0.01,
    ) -> np.ndarray:
        """Dynamically adjusts the radius to achieve a desired range of average neighbors per cell.

        Args:
            distances_matrix (np.ndarray): The precomputed distances matrix for the cluster cells.
            max_iterations (int): Maximum number of iterations to refine the radius. Defaults to 100.
            error_margin (float): Acceptable deviation from the target. Defaults to 7.5.
            initial_step_ratio (float): Initial step size as a fraction of the radius. Defaults to 0.1.

        Returns:
            np.ndarray: A matrix of stacked neighbours count with varying radius.

        Raises:
            RuntimeError: If a suitable radius cannot be found within the specified number of iterations.
        """
        # [Insert the revised function code here]
        distances_matrix_ = distances_matrix.copy()
        distances_matrix_[distances_matrix_ == 0.0] = np.nan
        radius = np.nanmean(distances_matrix_) * 1  # arbitrary
        step = radius * initial_step_ratio
        target_neighbors = 5  # arbitrary
        # target_neighbors = self.adata_prepared.uns["neighbors"]["params"]["n_neighbors"] / 10  # arbitrary
        previous_error = None

        result = []
        for iteration in range(max_iterations):
            neighbors_count = np.nansum(distances_matrix_ < radius, axis=1)
            avg_neighbors = np.nanmean(neighbors_count)
            error = target_neighbors - avg_neighbors
            result.append(neighbors_count)

            # print(f"Iteration {iteration + 1}: Radius = {radius:.4f}, Error = {error:.4f}, Step = {step:.4f}")
            # print(f"Target Neighbors: {target_neighbors}\nAverage Neighbors: {avg_neighbors}")

            if abs(error) <= error_margin:
                self.logger.info(
                    f"Suitable radius found: {radius:.4f} with average neighbors {avg_neighbors:.4f}, "
                    f"iteration {iteration + 1}."
                )
                return np.vstack(result)

            if previous_error is not None and np.sign(error) != np.sign(previous_error):
                step /= 2
                self.logger.debug(f"Overshoot detected. Reducing step size to {step:.6f}.")

            if error > 0:
                radius += step
            else:
                radius -= step

            if radius < 1e-6:
                radius = 1e-6
                self.logger.debug(f"Radius fell below minimum threshold. Setting radius to {radius}.")

            previous_error = error

        raise RuntimeError(
            f"Could not find a suitable radius within {max_iterations} iterations. "
            f"Final radius: {radius:.4f}, Final average neighbors: {avg_neighbors:.4f}."
        )
