#!/usr/bin/env python3

from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.stats import zscore

from sctram.infer._base import TrajectoryInferenceBase, x_diffmap_key, iroot_key


# TODO: label_key is already in TrajectoryInferenceBase

class DPTInference(TrajectoryInferenceBase):
    """DPT (Diffusion Pseudotime) trajectory inference method with customizable Diffusion Map calculation and root setting.

    This subclass of TrajectoryInferenceBase provides enhanced functionality for trajectory inference using the DPT method.
    It allows customization of the Diffusion Map calculation and provides flexible methods to set the root of the trajectory.

    Inherits From:
        TrajectoryInferenceBase: Provides base functionality for trajectory inference methods.

    Additional Features:
        - Customizable Diffusion Map calculation parameters.
        - Flexible root setting through various methods:
            - Based on Diffusion Map components.
            - Using a custom binary array.
            - Selecting a root based on cell labels.
    """

    def __init__(
        self,
        neighbors_params: Optional[Dict[str, Any]] = None,
        diffmap_params: Optional[Dict[str, Any]] = None,
        dpt_params: Optional[Dict[str, Any]] = None,
        iroot: Optional[Union[int, np.ndarray, dict]] = None,
        random_state: Optional[int] = None,
    ):
        """Initializes the DPT method with optional parameters for neighbors, Diffusion Map, DPT, and root selection.

        Args:
            neighbors_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.neighbors`.
            diffmap_params (Optional[Dict[str, Any]]): Parameters for `sc.tl.diffmap`.
            dpt_params (Optional[Dict[str, Any]]): Parameters for `sc.tl.dpt`.
            iroot (Optional[Union[int, np.ndarray, dict]]): Specification for the root of the trajectory.
                - If `int`, it represents the index of the root cell.
                - If `np.ndarray`, it should be a binary array with exactly one `1` indicating the root cell.
                - If `dict`, it represents a cell label from which the root will be selected. Dictionary must contain:
                    - `label_key` (str): The key in `adata.obs` that contains the labels.
                    - `label` (str): The specific label to select the root from.
                    - `method` (str): The method to select the root within the specified label. Supported methods:
                        - `'min_diffmap'`: Select the cell with the minimum value in a specified DiffMap component.
                        - `'max_diffmap'`: Select the cell with the maximum value in a specified DiffMap component.
                        - `'centroid'`: Select the cell closest to the centroid of the cluster.
                        - `'density'`: Select the cell in the cluster with the highest density.
                        - `'random'`: Select a random cell from the cluster.
                    - `outlier_definition_z` (float, optional): Decide whether or not ignore outliers in calculations.
                        If `None`, then all datapoints are taken into consideration.
                    - Additional method-specific parameters:
                        - For `'min_diffmap'` and `'max_diffmap'`:
                            - `component` (int, optional): The Diffusion Map component to use. Defaults to 0.
                        - For `centroid`:
                            - `centroid_embedding` (str): 'X' or a key from `adata.obsm` to be used to calculate 
                                the centroids. Defaults to 'X'.
                        - For `density`:
                            - `radius` (Optional[Union[float, str]]): To compute local density as the number of 
                                neighbors within a certain radius using `distances` matrix. If `None` or `"adjust"`, 
                                then the method will try to adjust it to make average density to be in [15, 30].
            random_state (Optional[int]): Random state for reproducibility.
        """
        super().__init__(
            neighbors_params=neighbors_params,
            method_params=dpt_params,
            random_state=random_state
        )
        self.diffmap_params = diffmap_params or {}
        self.iroot_spec = iroot  # Specification for the root

    def _calculate(self):
        """Performs the Diffusion Map and DPT calculations, including root setting."""
        # Compute Diffusion Map
        if x_diffmap_key not in self.adata_prepared.obsm:
            self.logger.info("Computing Diffusion Map.")
            sc.tl.diffmap(self.adata_prepared, **self.diffmap_params)
            self.logger.debug("Diffusion Map computed successfully.")
        else:
            self.logger.info("Diffusion Map is already calculated.")

        # Determine and set the root
        if self.iroot_spec is not None:
            if isinstance(self.iroot_spec, int):
                self.logger.debug(f"Setting root using provided integer index: {self.iroot_spec}.")
                self._set_root_custom_index(self.iroot_spec)
            elif isinstance(self.iroot_spec, np.ndarray):
                self.logger.debug("Setting root using provided custom binary array.")
                self.set_root_custom_array(custom_array = self.iroot_spec)
            elif isinstance(self.iroot_spec, dict):
                self.logger.debug("Setting root using provided label-based specification.")
                self.set_root_from_label_dict(label_dict = self.iroot_spec)
            else:
                raise ValueError("Invalid type for 'iroot'. Must be int, np.ndarray, or dict.")
        else:
            # Default root: cell with minimum value in the 4th Diffusion Map component
            self.logger.info("No root specification provided. Setting default root based on Diffusion Map.")
            self.set_root_from_diffmap()

        # Retrieve the root index
        if iroot_key not in self.adata_prepared.uns:
            raise RuntimeError("Root index ('iroot') has not been set.")

        # Perform DPT with the specified root in `iroot`
        self.logger.info("Performing DPT trajectory calculation.")
        sc.tl.dpt(self.adata_prepared, **self.method_params)
        self.logger.debug("DPT calculation completed successfully.")

    def get_result(self, return_mode: str) -> Union[AnnData, np.ndarray]:
        """Retrieves the result of the DPT trajectory inference.

        Args:
            return_mode (str): Decide the returned object. Either 'anndata' or 'vector'.
                - 'anndata': Returns the AnnData object with DPT results.
                - 'vector': Returns the pseudotime as a NumPy array.

        Raises:
            ValueError: If `return_mode` is invalid.
            RuntimeError: If trajectory inference has not been performed yet.

        Returns:
            Union[AnnData, np.ndarray]: The result of the DPT trajectory inference.
        """
        if self.adata_prepared is None:
            raise RuntimeError("First run `infer_trajectory`.")

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
            raise ValueError(f"Root index {root_ix} is out of bounds for the dataset with {self.adata_prepared.n_obs} cells.")
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
        """
        if not isinstance(custom_array, np.ndarray):
            raise TypeError("Custom root specification must be a NumPy array.")

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

        Returns:
            int: The index of the selected root cell.

        Raises:
            ValueError: If the specified label does not exist or method is invalid.
            RuntimeError: If Diffusion Map has not been computed.
        """
        root_ix, info = self._set_root_from_label_dict(label_dict=label_dict)
        self.adata_prepared.uns[iroot_key] = root_ix
        self.logger.info(info)
    
    def _set_root_from_label_dict(self, label_dict: Dict[str, Any]) -> Tuple[int, str]:
        """Method providing the root. See `set_root_from_label_dict` for details."""  # noqa
        required_keys = {'label_key', 'label', 'method', 'outlier_definition_z'}
        if not required_keys.issubset(label_dict.keys()):
            missing = required_keys - label_dict.keys()
            raise KeyError(f"Missing keys in label specification dictionary: {missing}")
        
        label_key = label_dict['label_key']
        label = label_dict['label']
        method = label_dict['method']
        outlier_definition_z = label_dict["outlier_definition_z"]
        method_params = {k: v for k, v in label_dict.items() if k not in required_keys}
        component_default = 0
        invalid_method_error = (
            "Invalid method for setting root. Choose from 'min_diffmap', "
            "'max_diffmap', 'centroid', 'density', or 'random'."
        )
        
        if label_key not in self.adata_prepared.obs:
            raise ValueError(f"Label key {label_key!r} not found in `adata.obs`.")

        if x_diffmap_key not in self.adata_prepared.obsm:
            raise RuntimeError("Diffusion Map has not been computed.")

        if label not in self.adata_prepared.obs[label_key].unique():
            raise ValueError(f"Label {label!r} not found in `adata.obs[{label_key!r}]`.")

        cluster_indices = np.where(self.adata_prepared.obs[label_key] == label)[0]
        if len(cluster_indices) == 0:
            raise ValueError(f"No cells found for label {label!r}.")

        master_indices = cluster_indices.copy()  # Initialize master_indices

        def _get_comp():
            _comp = method_params.get("component", component_default)
            if _comp < 0 or _comp >= self.adata_prepared.obsm[x_diffmap_key].shape[1]:
                raise ValueError(f"Component index {_comp} is out of bounds for Diffusion Map.")
            return _comp

        def _get_centroid_emb(subset):
            _centroid_emb = method_params.get("centroid_embedding", "X")
            if not (_centroid_emb == "X" or _centroid_emb in self.adata_prepared.obsm.keys()):
                raise ValueError(f"Centroid embedding key {_centroid_emb!r} is not found in `adata.obsm`.")
            _embedding = self.adata_prepared.X if _centroid_emb == "X" else self.adata_prepared.obsm[_centroid_emb]
            return _embedding[subset]

        if outlier_definition_z is not None:
            if method in ['min_diffmap', 'max_diffmap', 'density']:
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
                            f"All cells in label '{label}' were identified as outliers based on z-score threshold "
                            f"{outlier_definition_z}. Proceeding with all cluster cells."
                        )
                        master_indices = cluster_indices
                    else:
                        master_indices = refined_indices
                        self.logger.info(
                            f"Outlier removal applied. {len(master_indices)} out of {len(cluster_indices)} cells remain "
                            f"after filtering based on z-score <= {outlier_definition_z}."
                        )

            elif method == 'centroid':
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
                        f"All cells in label '{label}' were identified as outliers based on z-score threshold "
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
            self.logger.info(f"No outlier removal applied. Using all {len(master_indices)} cells in label '{label}'.")


        if method == 'min_diffmap':
            component = _get_comp()
            cluster_diffmap = self.adata_prepared.obsm[x_diffmap_key][master_indices, component]
            root_sub_ix = np.argmin(cluster_diffmap)
            root_ix = master_indices[root_sub_ix]
            info = f"Root set to cell index {root_ix} based on minimum in DiffMap component {component} within label '{label}'."
            return root_ix, info

        elif method == 'max_diffmap':
            component = _get_comp()
            cluster_diffmap = self.adata_prepared.obsm[x_diffmap_key][master_indices, component]
            root_sub_ix = np.argmax(cluster_diffmap)
            root_ix = master_indices[root_sub_ix]
            info = f"Root set to cell index {root_ix} based on maximum in DiffMap component {component} within label '{label}'."
            return root_ix, info

        elif method == 'centroid':
            embedding = _get_centroid_emb(subset=master_indices)  # shape: (n_master_cells, n_features)
            centroid = np.mean(embedding, axis=0)
            distances = np.linalg.norm(embedding - centroid, axis=1)
            root_sub_ix = np.argmin(distances)
            root_ix = master_indices[root_sub_ix]
            info = f"Root set to cell index {root_ix} closest to centroid within label '{label}'."
            return root_ix, info

        elif method == 'density':
            if "distances" not in self.adata_prepared.obsp:
                raise RuntimeError(
                    "Distances matrix has not been computed. Cannot compute density. Call scanpy `neighbor` method."
                )
            distances_matrix = self.adata_prepared.obsp['distances'][np.ix_(master_indices, master_indices)].toarray()

            # Compute local density as the number of neighbors within a certain radius
            # Currently using a fixed radius; this can be parameterized as needed.
            radius = method_params.get("radius", None)
            if radius is None or radius == "adjust":
                radius = self._adjust_radius_for_density(distances_matrix, max_iterations=100)

            # Extract the distances for the master_indices
            density = np.sum(distances_matrix < radius, axis=1)
            if np.all(density == 0):
                self.logger.warning(
                    f"No neighbors found within radius {radius} for any cell in label '{label}'. "
                    "Density-based root selection may not be meaningful."
                )
            root_sub_ix = np.argmin(density)
            root_ix = master_indices[root_sub_ix]
            info = f"Root set to cell index {root_ix} with highest density within label '{label}'."
            return root_ix, info

        elif method == 'random':
            rng = np.random.default_rng(self.random_state)
            root_ix = rng.choice(master_indices)
            info = f"Root set to randomly selected cell index {root_ix} within label '{label}'."
            return root_ix, info

        else:
            raise ValueError(invalid_method_error)

    def _adjust_radius_for_density(
        self, 
        distances_matrix: np.ndarray, 
        max_iterations: int = 100, 
        error_margin: float = 3, 
        initial_step_ratio: float = 0.1
    ) -> float:
        """Dynamically adjusts the radius to achieve a desired range of average neighbors per cell.

        Args:
            distances_matrix (np.ndarray): The precomputed distances matrix for the cluster cells.
            max_iterations (int): Maximum number of iterations to refine the radius. Defaults to 100.
            target_neighbors (float): Target average number of neighbors. Defaults to 22.5.
            error_margin (float): Acceptable deviation from the target. Defaults to 7.5.
            initial_step_ratio (float): Initial step size as a fraction of the radius. Defaults to 0.1.

        Returns:
            float: Adjusted radius that yields an average number of neighbors within the target range.

        Raises:
            RuntimeError: If a suitable radius cannot be found within the specified number of iterations.
        """
        distances_matrix_ = distances_matrix.copy()
        distances_matrix_[distances_matrix_ == 0.0] = np.nan
        radius = np.nanmean(distances_matrix_) * 3  # Initial guess for the radius
        step = radius * initial_step_ratio  # Initial step size based on the radius
        target_neighbors = self.adata_prepared.uns["neighbors"]["params"]["n_neighbors"] / 4

        for iteration in range(max_iterations):
            
            neighbors_count = distances_matrix.shape[0] - np.sum(distances_matrix < radius, axis=1)
            avg_neighbors = np.mean(neighbors_count)
            error = target_neighbors - avg_neighbors

            print(f"Iteration {iteration + 1}: Radius = {radius:.4f}, Error = {error:.4f}, Step = {step:.4f}")
            print(target_neighbors)
            print(avg_neighbors)
    
            # Check if the current average is within the acceptable range
            if abs(error) <= error_margin:
                self.logger.info(
                    f"Suitable radius found: {radius:.4f} with average neighbors {avg_neighbors:.4f}, "
                    f"iteration {iteration + 1}."
                )
                return radius

            # Adjust the radius based on the error direction
            radius_ = radius.copy()
            if error > 0:  # Need more neighbors: increase radius
                radius -= step
            else:  # Need fewer neighbors: decrease radius
                radius += step
                
            if radius < 1e-6:
                radius = radius_.copy()  # Ensure radius remains positive
                step = radius * initial_step_ratio
                print("Here")
            
        raise RuntimeError(
            f"Could not find a suitable radius within {max_iterations} iterations. "
            f"Final radius: {radius:.4f}, Final average neighbors: {avg_neighbors:.4f}."
        )
