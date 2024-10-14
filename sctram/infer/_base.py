#!/usr/bin/env python3

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from sctram._constants import connectivities_key, distances_key, labels_key, neighbors_key, x_diffmap_key


class AnndataPreperation:
    """Mixin class for preparing anndata from different input types.

    This class includes common initialization and data preparation steps.
    Subclasses should implement the specific calculation and return logic.

    This class supports three initialization methods:

        1. **AnnData with Precomputed Neighbors**:
            - `adata`: An `AnnData` object containing the data.
            - `neighbour_key`: The key in `adata.uns` where precomputed neighbors are stored.
            - `labels`: (Optional) Labels if not present in `adata.obs`.

        2. **Raw Embedding and Labels with Optional Precomputed Neighbors**:
            - `embedding`: Raw embedding data (`numpy.ndarray` or `pandas.DataFrame`).
            - `labels`: Corresponding labels (`numpy.ndarray` or `pandas.Series`).
            - `connectivities`: (Optional) Connectivity matrix (`numpy.ndarray` or `pandas.DataFrame`).
            - `distances`: (Optional) Distance matrix (`numpy.ndarray` or `pandas.DataFrame`).

        3. **AnnData without Precomputed Neighbors**:
            - `adata`: An `AnnData` object without precomputed neighbors.
            - `labels`: (Optional) Labels if not present in `adata.obs`.

    **Note**: Use only one initialization method per instance.
    """

    def __init__(self, random_state) -> None:
        """Initialize object.

        Args:
            random_state (Optional[int], optional): See `TrajectoryEmbeddingBase.__init__`.
        """
        self.logger = logging.getLogger(self.__class__.__name__)  # Configure logging
        self.random_state = random_state

    def initialize_adata(
        self,
        adata: Optional[AnnData],
        embedding: Optional[Union[np.ndarray, pd.DataFrame]],
        labels: Optional[Union[np.ndarray, pd.Series]],
        connectivities: Optional[Union[np.ndarray, pd.DataFrame]],
        distances: Optional[Union[np.ndarray, pd.DataFrame]],
        neighbour_key: Optional[str],
    ) -> AnnData:
        """Initializes the AnnData object based on provided inputs.

        This includes handling different initialization methods.

        Args:
            adata (Optional[AnnData]): An AnnData object.
            embedding (Optional[Union[np.ndarray, pd.DataFrame]]): Embedding data.
            labels (Optional[Union[np.ndarray, pd.Series]]): Labels.
            connectivities (Optional[Union[np.ndarray, pd.DataFrame]]): Connectivity matrix.
            distances (Optional[Union[np.ndarray, pd.DataFrame]]): Distance matrix.
            neighbour_key (Optional[str]): Key for precomputed neighbors.

        Returns:
            AnnData: Prepared AnnData object.

        Raises:
            ValueError: If input validation fails.
        """
        input_methods = {
            "adata_with_neighbors": adata is not None and neighbour_key is not None,
            "adata_without_neighbors": adata is not None and neighbour_key is None,
            "embedding_labels": embedding is not None and labels is not None,
        }

        methods_used = sum(input_methods.values())

        if methods_used == 0:
            raise ValueError("No valid input provided. Provide either an AnnData object or both embedding and labels.")
        elif methods_used > 1:
            raise ValueError("Multiple input methods provided. Please provide inputs using only one method at a time.")

        if input_methods["adata_with_neighbors"]:
            adata_prepared = self._initialize_from_adata_with_neighbors(
                adata=adata, labels=labels, neighbour_key=neighbour_key
            )
        elif input_methods["adata_without_neighbors"]:
            adata_prepared = self._initialize_from_adata_without_neighbors(adata=adata, labels=labels)
        elif input_methods["embedding_labels"]:
            adata_prepared = self._initialize_from_embedding_labels(
                embedding=embedding, labels=labels, connectivities=connectivities, distances=distances
            )
        else:
            raise ValueError("Invalid initialization method.")

        return adata_prepared

    def _initialize_from_adata_with_neighbors(
        self,
        adata: AnnData,
        labels: Optional[Union[np.ndarray, pd.Series]],
        neighbour_key: Optional[str],
    ) -> AnnData:
        """Initializes AnnData from an existing AnnData with precomputed neighbors.

        Args:
            adata (AnnData): Input AnnData object.
            labels (Optional[Union[np.ndarray, pd.Series]]): Labels to add to `adata.obs['labels']`.
            neighbour_key (Optional[str]): Key in `adata.uns` where precomputed neighbors are stored.

        Returns:
            AnnData: Prepared AnnData object.

        Raises:
            ValueError: If input validation fails.
        """
        self.logger.debug("Initializing from AnnData with precomputed neighbors.")
        adata_prepared = adata.copy()

        # Handle labels
        adata_prepared = self._add_labels_to_adata(adata_prepared, labels)

        # Handle precomputed neighbors
        if neighbour_key:
            self.logger.debug(f"Using precomputed neighbors from key {neighbour_key!r}.")
            if neighbour_key not in adata_prepared.uns:
                raise ValueError(f"Neighbours key {neighbour_key!r} not found in AnnData.uns.")
            # Move neighbors to standard neighbors_key key
            adata_prepared.uns[neighbors_key] = adata_prepared.uns.pop(neighbour_key)
        else:
            if neighbors_key not in adata_prepared.uns:
                self.logger.warning("No precomputed neighbors provided.")

        self.logger.debug("AnnData initialized successfully from AnnData with neighbors.")
        return adata_prepared

    def _initialize_from_adata_without_neighbors(
        self,
        adata: AnnData,
        labels: Optional[Union[np.ndarray, pd.Series]],
    ) -> AnnData:
        """Initializes AnnData from an existing AnnData without precomputed neighbors.

        Args:
            adata (AnnData): Input AnnData object.
            labels (Optional[Union[np.ndarray, pd.Series]]): Labels to add to `adata.obs['labels']`.

        Returns:
            AnnData: Prepared AnnData object.
        """
        self.logger.debug("Initializing from AnnData without precomputed neighbors.")
        adata_prepared = adata.copy()

        # Handle labels
        adata_prepared = self._add_labels_to_adata(adata_prepared, labels)

        if neighbors_key not in adata_prepared.uns:
            self.logger.info("No precomputed neighbors found in AnnData.")
        else:
            self.logger.debug("Using existing neighbors from AnnData.")

        self.logger.debug("AnnData initialized successfully from AnnData without neighbors.")
        return adata_prepared

    def _initialize_from_embedding_labels(
        self,
        embedding: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        connectivities: Union[np.ndarray, pd.DataFrame],
        distances: Union[np.ndarray, pd.DataFrame],
    ) -> AnnData:
        """Initializes AnnData from raw embedding and labels, with optional neighbors.

        Args:
            embedding (Union[np.ndarray, pd.DataFrame]): Low-dimensional representation.
            labels (Union[np.ndarray, pd.Series]): Cell labels.
            connectivities (Optional[Union[np.ndarray, pd.DataFrame]]): Connectivity matrix.
            distances (Optional[Union[np.ndarray, pd.DataFrame]]): Distance matrix.

        Returns:
            AnnData: Prepared AnnData object.
        """
        self.logger.debug("Initializing from embedding and labels.")
        adata = self._create_anndata_from_embedding(embedding, labels)

        # Handle precomputed neighbors
        if connectivities is not None and distances is not None:
            self.logger.debug("Adding precomputed neighbors to AnnData.")
            adata = self._add_precomputed_neighbors(adata, connectivities=connectivities, distances=distances)
        else:
            self.logger.info("Precomputed neighbor matrices (`distances` and `connectivities`) are not provided.")

        self.logger.debug("AnnData initialized successfully from embedding and labels.")
        return adata

    def _add_labels_to_adata(self, adata: AnnData, labels: Optional[Union[np.ndarray, pd.Series]]) -> AnnData:
        """Adds labels to the AnnData object.

        Args:
            adata (AnnData): AnnData object.
            labels (Optional[Union[np.ndarray, pd.Series]]): Labels to add.

        Returns:
            AnnData: Updated AnnData object.

        Raises:
            ValueError: If input validation fails.
        """
        if labels is not None:
            self.logger.debug("Adding provided labels to AnnData object.")
            if isinstance(labels, (pd.Series, pd.DataFrame)):
                labels = labels.squeeze()
            elif isinstance(labels, np.ndarray):
                labels = pd.Series(labels, index=adata.obs_names)
            else:
                raise ValueError("Labels must be a numpy.ndarray or pandas.Series/DataFrame.")

            if len(adata) != len(labels):
                raise ValueError("The number of cells in AnnData and labels must match.")

            adata.obs[labels_key] = labels.values
        else:
            if labels_key not in adata.obs:
                raise ValueError("Labels not found in AnnData object. Please provide labels separately.")
            self.logger.debug("Using existing labels from AnnData object.")

        return adata

    def _create_anndata_from_embedding(
        self,
        embedding: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
    ) -> AnnData:
        """Creates an AnnData object from embedding and labels.

        Args:
            embedding (Union[np.ndarray, pd.DataFrame]): Low-dimensional representation.
            labels (Union[np.ndarray, pd.Series]): Cell labels.

        Returns:
            AnnData: Created AnnData object.

        Raises:
            ValueError: If input validation fails.
        """
        self.logger.debug("Creating AnnData object from embedding.")
        if isinstance(embedding, pd.DataFrame):
            adata = AnnData(embedding.values)
            adata.obs_names = embedding.index
            adata.var_names = [f"Dim{i}" for i in range(embedding.shape[1])]
        elif isinstance(embedding, np.ndarray):
            adata = AnnData(embedding)
            adata.var_names = [f"Dim{i}" for i in range(embedding.shape[1])]
        else:
            raise ValueError("Embedding must be a numpy.ndarray or pandas.DataFrame.")

        # Convert labels to a pandas Series
        if isinstance(labels, (pd.Series, pd.DataFrame)):
            labels = labels.squeeze()
        elif isinstance(labels, np.ndarray):
            labels = pd.Series(labels, index=adata.obs_names)
        else:
            raise ValueError("Labels must be a numpy.ndarray or pandas.Series/DataFrame.")

        if len(adata) != len(labels):
            raise ValueError("The number of embeddings and labels must match.")

        adata.obs[labels_key] = labels.values

        self.logger.debug("AnnData object created successfully from embedding and labels.")
        return adata

    def _add_precomputed_neighbors(
        self,
        adata: AnnData,
        connectivities: Union[np.ndarray, pd.DataFrame],
        distances: Union[np.ndarray, pd.DataFrame],
    ) -> AnnData:
        """Adds precomputed neighbors to the AnnData object.

        Ensures that `adata.uns[neighbors_key]` has the required structure:
            {
                'connectivities_key': connectivities_key,
                'distances_key': distances_key,
                'params': {
                    'n_neighbors': <inferred_n_neighbors>
                }
            }

        Args:
            adata (AnnData): AnnData object.
            connectivities (Optional[Union[np.ndarray, pd.DataFrame]]): Connectivity matrix.
            distances (Optional[Union[np.ndarray, pd.DataFrame]]): Distance matrix.

        Returns:
            AnnData: Updated AnnData object.

        Raises:
            ValueError: If input validation fails.
        """
        self.logger.debug("Adding connectivity matrix to AnnData.")
        if isinstance(connectivities, pd.DataFrame):
            connectivities = connectivities.values
        if not isinstance(connectivities, (np.ndarray, csr_matrix)):
            raise ValueError("Connectivities must be a numpy.ndarray, pandas.DataFrame, or scipy.sparse matrix.")
        if connectivities.shape[0] != connectivities.shape[1]:
            raise ValueError("Connectivity matrix must be square.")
        if connectivities.shape[0] != len(adata):
            raise ValueError("Connectivity matrix size must match number of cells.")
        adata.obsp[connectivities_key] = connectivities

        self.logger.debug("Adding distance matrix to AnnData.")
        if isinstance(distances, pd.DataFrame):
            distances = distances.values
        if not isinstance(distances, (np.ndarray, csr_matrix)):
            raise ValueError("Distances must be a numpy.ndarray, pandas.DataFrame, or scipy.sparse matrix.")
        if distances.shape[0] != distances.shape[1]:
            raise ValueError("Distance matrix must be square.")
        if distances.shape[0] != len(adata):
            raise ValueError("Distance matrix size must match number of cells.")
        adata.obsp[distances_key] = distances

        if connectivities.shape[0] != distances.shape[1]:
            raise ValueError("Connectivity and distance matrices must be the same shape.")

        # Infer n_neighbors from the distances and connectivities matrices
        n_neighbors_inferred = self._infer_n_neighbors(distances, connectivities)
        adata.uns[neighbors_key] = {  # Update adata.uns[neighbors_key] with the required structure
            "connectivities_key": connectivities_key,
            "distances_key": distances_key,
            "params": {
                "n_neighbors": n_neighbors_inferred
                # Exclude 'metric', 'random_state', 'method' as per requirements
            },
        }
        for adata_uns_neighbors_params_keys in ["metric", "random_state", "method"]:
            adata.uns[neighbors_key]["params"][adata_uns_neighbors_params_keys] = np.nan

        self.logger.debug(
            "Precomputed neighbors added successfully with the required structure in adata.uns[neighbors_key]."
        )
        return adata

    def _infer_n_neighbors(self, distances, connectivities, sample_size: int = 10) -> int:
        """Infers the number of neighbors (n_neighbors) by ensuring both connectivity and distance matrices agree.

        Args:
            distances (Union[np.ndarray, pd.DataFrame, csr_matrix]): Distance matrix.
            connectivities (Union[np.ndarray, pd.DataFrame, csr_matrix]): Connectivity matrix.
            sample_size (int): Number of random rows to sample for consistency check.

        Returns:
            int: Inferred number of neighbors.

        Raises:
            ValueError: If connectivity and distance matrices do not agree on sampled rows.
        """
        self.logger.debug("Starting n_neighbors inference.")

        # Helper function to count non-zero elements per row
        def count_nonzeros(matrix, matrix_name):
            if isinstance(matrix, csr_matrix):
                return matrix.getnnz(axis=1)
            elif isinstance(matrix, np.ndarray):
                if matrix_name == "connectivities":
                    return np.sum(matrix > 0, axis=1)
                elif matrix_name == "distances":
                    return np.sum(np.isfinite(matrix), axis=1)
            elif isinstance(matrix, pd.DataFrame):
                return matrix.astype(bool).sum(axis=1).values
            else:
                raise ValueError(f"Unsupported matrix type for {matrix_name}.")

        # Count non-zero neighbors for both matrices
        nnz_connectivities = count_nonzeros(connectivities, "connectivities")
        nnz_distances = count_nonzeros(distances, "distances")

        # Sample indices
        rng = np.random.default_rng(self.random_state)
        sample_size = min(sample_size, len(nnz_connectivities))
        sampled_indices = rng.choice(len(nnz_connectivities), size=sample_size, replace=False)
        self.logger.debug(f"Sampled indices for consistency check: {sampled_indices}")

        # Compare neighbor counts in sampled rows
        for idx in sampled_indices:
            conn_count = nnz_connectivities[idx]
            dist_count = nnz_distances[idx]
            if conn_count != dist_count:
                raise ValueError(
                    f"Neighbor count mismatch between connectivities and distances matrices at sampled row {idx}."
                )

        # If all sampled rows match, infer n_neighbors from connectivity matrix
        # Assuming connectivity matrix defines a KNN graph with consistent n_neighbors
        unique_conn_counts = np.unique(nnz_connectivities).max()
        unique_dist_counts = np.unique(nnz_distances).max()
        if unique_dist_counts != unique_dist_counts:
            raise ValueError("Max neighbor count mismatch between connectivities and distances matrices.")

        return int(unique_conn_counts)


class InferenceAndEmbeddingBase(ABC):
    """Abstract base class for trajectory inference methods or embedding calculations.

    This class includes common initialization and data preparation steps.
    Subclasses should implement the specific calculation and return logic.

    This class supports three initialization methods given in `AnndataBase`.
    """

    subclass_mode: Literal["trajectory inference", "embedding calculation"]

    def __init__(
        self,
        random_state: Optional[int] = None,
        # adata preparation parameters
        adata: Optional[AnnData] = None,
        embedding: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        labels: Optional[Union[np.ndarray, pd.Series]] = None,
        connectivities: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        distances: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        neighbour_key: Optional[str] = None,
    ):
        """Initializes the class and prepares the AnnData object.

        See `AnndataPreperationMixin` class to understand how to use anndata parameters to prepare the anndata object.

        Args:
            random_state (Optional[int]): Random state for reproducibility.
            adata (Optional[AnnData]): An AnnData object containing the dataset. It can be used directly
                if it contains precomputed neighbors or will require neighbor computation if not.
            embedding (Optional[Union[np.ndarray, pd.DataFrame]]): A low-dimensional representation of the data,
                resulting from dimensionality reduction techniques such as PCA, or an autoencoder latent space.
            labels (Optional[Union[np.ndarray, pd.Series]]): The labels or identifiers for each data point in
                the embedding. These are used to group data points during the PAGA computation.
            connectivities (Optional[Union[np.ndarray, pd.DataFrame]]): A precomputed connectivity matrix specifying the
                connectivity relationships between data points, used if neighbors are not computed within this function.
            distances (Optional[Union[np.ndarray, pd.DataFrame]]): A precomputed distance matrix specifying distances.
                between data points, which can be used alongside connectivities to define neighborhood relationships.
            neighbour_key (Optional[str]): The key under which precomputed neighbors are stored within the `adata.uns`
                dictionary if using an AnnData object that already contains neighbor information.

        Raises:
            ValueError: if `subclass_mode` is not correctly defined among possible ones.
            NotImplementedError: When `subclass_mode` is not specified as class variable.
        """
        if not hasattr(self, "subclass_mode"):
            raise NotImplementedError("Subclasses must define the 'subclass_mode' class variable")
        if self.subclass_mode not in ["trajectory inference", "embedding calculation"]:
            raise ValueError("'subclass_mode' must be 'trajectory inference' or 'embedding calculation'")

        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)  # Configure logging

        if self.random_state is not None:  # Set random seed
            sc.settings.seed = self.random_state
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # Initialize AnnData
        self.adata_prepared = AnndataPreperation(random_state=self.random_state).initialize_adata(
            adata=adata,
            embedding=embedding,
            labels=labels,
            connectivities=connectivities,
            distances=distances,
            neighbour_key=neighbour_key,
        )

    def calculate(self, return_mode: Optional[str] = None) -> Any:
        """Computes the trajectory inference or embedding calculation based on the provided inputs.

        Args:
            return_mode (Optional[str]): Decide the returned object. Either anndata or the result of the calculation.
                The key `anndata` used as for outputing anndata. Other keys are calculation specific. When `None`,
                nothing is returned. The user needs to call `get_result` separately.

        Returns:
            Any: The result of the specific trajectory inference method or embedding calculation.

        Raises:
            RuntimeError: If fails.
        """
        try:
            self.logger.info(f"Starting {self.__class__.__name__} {self.subclass_mode}.")

            # Perform the specific calculation
            self.logger.info(f"Performing {self.subclass_mode}.")
            self._calculate()

            # Return the results based on the subclass's implementation
            self.logger.info(f"{self.subclass_mode.capitalize()} completed successfully.")
            if return_mode is not None:
                return self.get_result(return_mode=return_mode)
                # Method specific output or outputs.

        except Exception as e:
            self.logger.error(f"Error in {self.subclass_mode}: {e}")
            raise RuntimeError(f"Error in {self.subclass_mode}: {e}") from e

    def _needs_neighbors(self, neighbors_params):
        if neighbors_key not in self.adata_prepared.uns:
            self.logger.info("Computing neighbors.")
            sc.pp.neighbors(self.adata_prepared, **neighbors_params)
        else:
            self.logger.info("Using precomputed neighbors from AnnData.")

    def _needs_diffmap(self, diffmap_params):
        if x_diffmap_key not in self.adata_prepared.obsm:
            self.logger.info("Computing Diffusion Map.")
            sc.tl.diffmap(self.adata_prepared, **diffmap_params)
            self.logger.debug("Diffusion Map computed successfully.")
        else:
            self.logger.info("Diffusion Map is already calculated.")

    @abstractmethod
    def _calculate(self):
        """Performs the specific trajectory inference or embedding calculation."""
        # Specify if you need to call `_needs_neighbors` method.
        pass

    @abstractmethod
    def get_result(self, return_mode: str) -> Any:
        """Retrieves the result of the trajectory inference or embedding calculation.

        Args:
            return_mode (str): Decide the returned object. Either anndata or the result of the calculation. The key
                `anndata` used to get the anndata with calculations. Other keys are calculation specific.
        """
        pass


class EmbeddingBase(InferenceAndEmbeddingBase):
    """Embedding subclass."""

    subclass_mode = "embedding calculation"


class InferenceBase(InferenceAndEmbeddingBase):
    """Inference subclass."""

    subclass_mode = "trajectory inference"
