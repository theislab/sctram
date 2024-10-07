#!/usr/bin/env python3

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


class TrajectoryInferenceBase(ABC):
    """Abstract base class for trajectory inference methods.

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

    def __init__(
        self,
        neighbors_params: Optional[Dict[str, Any]] = None,
        method_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ):
        """Initializes the base trajectory inference method with common parameters.

        Args:
            neighbors_params (Optional[Dict[str, Any]]): Parameters for `sc.pp.neighbors`.
            method_params (Optional[Dict[str, Any]]): Parameters for the specific trajectory method.
            random_state (Optional[int]): Random state for reproducibility.
        """
        self.neighbors_params = neighbors_params or {}
        self.method_params = method_params or {}
        self.random_state = random_state

        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Prevent adding multiple handlers in interactive environments
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Internal AnnData object
        self.adata_prepared: Optional[AnnData] = None

    def infer_trajectory(
        self,
        adata: Optional[AnnData] = None,
        embedding: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        labels: Optional[Union[np.ndarray, pd.Series]] = None,
        connectivities: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        distances: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        neighbour_key: Optional[str] = None,
        return_mode: Optional[str] = None,
    ) -> Any:
        """Computes the trajectory inference based on the provided inputs.

        This method prepares the AnnData object and calls the specific calculation.

        Args:
            adata (Optional[AnnData]): An AnnData object containing the dataset. It can be used directly
                if it contains precomputed neighbors or will require neighbor computation if not.
            embedding (Optional[Union[np.ndarray, pd.DataFrame]]): A low-dimensional representation of the data,
                typically resulting from dimensionality reduction techniques such as PCA, or an autoencoder latent space.
            labels (Optional[Union[np.ndarray, pd.Series]]): The labels or identifiers for each data point in
                the embedding. These are used to group data points during the PAGA computation.
            connectivities (Optional[Union[np.ndarray, pd.DataFrame]]): A precomputed connectivity matrix specifying the
                connectivity relationships between data points, used if neighbors are not computed within this function.
            distances (Optional[Union[np.ndarray, pd.DataFrame]]): A precomputed distance matrix specifying the distances
                between data points, which can be used alongside connectivities to define neighborhood relationships.
            neighbour_key (Optional[str]): The key under which precomputed neighbors are stored within the `adata.uns`
                dictionary if using an AnnData object that already contains neighbor information.
            return_mode (Optional[str]): Decide the returned object. Either anndata or the result of the calculation.
                The key `anndata` used as for outputing anndata. Other keys are calculation specific. When `None`,
                nothing is returned. The user needs to call `get_result` separately.

        Returns:
            Any: The result of the specific trajectory inference method.

        Raises:
            RuntimeError: If trajectory inference fails.
        """
        try:
            self.logger.info(f"Starting {self.__class__.__name__} trajectory inference.")

            # Initialize AnnData
            self.adata_prepared = self._initialize_adata(
                adata=adata,
                embedding=embedding,
                labels=labels,
                connectivities=connectivities,
                distances=distances,
                neighbour_key=neighbour_key,
            )

            # Set random state if provided
            if self.random_state is not None:
                sc.settings.seed = self.random_state

            # Ensure neighbors are computed
            if "neighbors" not in self.adata_prepared.uns:
                self.logger.info("Computing neighbors.")
                sc.pp.neighbors(self.adata_prepared, **self.neighbors_params)
            else:
                self.logger.info("Using precomputed neighbors from AnnData.")

            # Perform the specific trajectory calculation
            self.logger.info("Performing trajectory calculation.")
            self._calculate()

            # Return the results based on the subclass's implementation
            self.logger.info("Trajectory inference completed successfully.")
            if return_mode is not None:
                return self.get_result(return_mode=return_mode)
                # Method specific output or outputs.

        except Exception as e:
            self.logger.error(f"Error in trajectory inference: {e}")
            raise RuntimeError(f"Error in trajectory inference: {e}") from e

    def _initialize_adata(
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
            # Move neighbors to standard 'neighbors' key
            adata_prepared.uns["neighbors"] = adata_prepared.uns.pop(neighbour_key)
        else:
            if "neighbors" not in adata_prepared.uns:
                self.logger.warning("No precomputed neighbors provided. Neighbors will be computed.")

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

        if "neighbors" not in adata_prepared.uns:
            self.logger.info("No precomputed neighbors found in AnnData. Neighbors will be computed.")
        else:
            self.logger.debug("Using existing neighbors from AnnData.")

        self.logger.debug("AnnData initialized successfully from AnnData without neighbors.")
        return adata_prepared

    def _initialize_from_embedding_labels(
        self,
        embedding: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        connectivities: Optional[Union[np.ndarray, pd.DataFrame]],
        distances: Optional[Union[np.ndarray, pd.DataFrame]],
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
        if connectivities is not None or distances is not None:
            self.logger.debug("Adding precomputed neighbors to AnnData.")
            adata = self._add_precomputed_neighbors(adata, connectivities=connectivities, distances=distances)
        else:
            self.logger.info("No precomputed neighbors provided. Neighbors will be computed.")

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

            adata.obs["labels"] = labels.values
        else:
            if "labels" not in adata.obs:
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

        adata.obs["labels"] = labels.values

        self.logger.debug("AnnData object created successfully from embedding and labels.")
        return adata

    def _add_precomputed_neighbors(
        self,
        adata: AnnData,
        connectivities: Optional[Union[np.ndarray, pd.DataFrame]],
        distances: Optional[Union[np.ndarray, pd.DataFrame]],
    ) -> AnnData:
        """Adds precomputed neighbors to the AnnData object.

        Args:
            adata (AnnData): AnnData object.
            connectivities (Optional[Union[np.ndarray, pd.DataFrame]]): Connectivity matrix.
            distances (Optional[Union[np.ndarray, pd.DataFrame]]): Distance matrix.

        Returns:
            AnnData: Updated AnnData object.

        Raises:
            ValueError: If input validation fails.
        """
        if connectivities is not None:
            self.logger.debug("Adding connectivity matrix to AnnData.")
            if isinstance(connectivities, pd.DataFrame):
                connectivities = connectivities.values
            if not isinstance(connectivities, np.ndarray):
                raise ValueError("Connectivities must be a numpy.ndarray or pandas.DataFrame.")

            if connectivities.shape[0] != connectivities.shape[1]:
                raise ValueError("Connectivity matrix must be square.")
            if connectivities.shape[0] != len(adata):
                raise ValueError("Connectivity matrix size must match number of cells.")

            adata.obsp["connectivities"] = connectivities

        if distances is not None:
            self.logger.debug("Adding distance matrix to AnnData.")
            if isinstance(distances, pd.DataFrame):
                distances = distances.values
            if not isinstance(distances, np.ndarray):
                raise ValueError("Distances must be a numpy.ndarray or pandas.DataFrame.")

            if distances.shape[0] != distances.shape[1]:
                raise ValueError("Distance matrix must be square.")
            if distances.shape[0] != len(adata):
                raise ValueError("Distance matrix size must match number of cells.")

            adata.obsp["distances"] = distances
        else:
            if "distances" not in adata.obsp:
                self.logger.warning("Distances not provided. Creating a placeholder distance matrix.")
                adata.obsp["distances"] = np.zeros((len(adata), len(adata)))

        # Ensure both connectivities and distances are present
        if "connectivities" not in adata.obsp:
            self.logger.warning("Connectivities not provided. Creating a placeholder connectivity matrix.")
            adata.obsp["connectivities"] = np.ones((len(adata), len(adata)))  # Placeholder

        if "distances" not in adata.obsp:
            self.logger.warning("Distances not provided. Creating a placeholder distance matrix.")
            adata.obsp["distances"] = np.zeros((len(adata), len(adata)))  # Placeholder

        self.logger.debug("Precomputed neighbors added successfully.")
        return adata

    @abstractmethod
    def _calculate(self):
        """Performs the specific trajectory inference calculation."""
        pass

    @abstractmethod
    def get_result(self, return_mode: str) -> Any:
        """Retrieves the result of the trajectory inference.

        Args:
            return_mode (str): Decide the returned object. Either anndata or the result of the calculation. The key
                `anndata` used to get the anndata with calculations. Other keys are calculation specific.
        """
        pass
