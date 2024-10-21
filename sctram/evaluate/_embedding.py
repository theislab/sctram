#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import procrustes
from sklearn.manifold import Isomap

from sctram.evaluate._base import EvaluationBase
from sctram.evaluate._metricsmixin._embeddingmetricsmixin import EmbeddingMetricsMixin


class EmbeddingEvaluation(EmbeddingMetricsMixin, EvaluationBase):
    """Evaluation method to compare inferred embeddings with a given trajectory.

    This class compares the inferred embeddings (numpy array of shape [n_samples, n_components])
    with the given trajectory (networkx.MultiDiGraph), using various metrics to assess how well
    the embedding captures the trajectory structure.

    Metrics include Procrustes distance, correlation measures, mean squared error, cosine similarity,
    topological metrics like Moran's I, Geary's C, Local Moran's I (LISA), Getis-Ord Gi*, and more.
    """

    def __init__(
        self,
        method_params: Dict[str, Any],
        subset_params: Optional[Dict[str, Any]] = None,
        prepare_params_before_subset: Optional[Dict[str, Any]] = None,
        prepare_params_after_subset: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the Embedding evaluation method.

        Args:
            method_params (Dict[str, Any]): Parameters specific to the evaluation method.
                - 'embedding_method': One of ['pca', 'diffmap', 'umap', 'gene_space'].
                - Other method-specific parameters as needed.
            subset_params (Optional[Dict[str, Any]]): Parameters for subsetting the data.
            prepare_params_before_subset (Optional[Dict[str, Any]]): Parameters for preparing the data before subsetting.
            prepare_params_after_subset (Optional[Dict[str, Any]]): Parameters for preparing the data after subsetting.

        Raises:
            ValueError: Must have keys not in `prepare_params_before_subset`.
        """
        super().__init__(
            method_params=method_params,
            subset_params=subset_params,
            prepare_params_before_subset=prepare_params_before_subset,
            prepare_params_after_subset=prepare_params_after_subset,
        )
        self.logger.debug(f"Initialized EmbeddingEvaluation with metrics: {self.metrics}")

        if prepare_params_before_subset is None or "method" not in prepare_params_before_subset:
            raise ValueError("Determine a method how to compute reference distances from the given trajectory.")

    def get_result(self) -> Any:
        """Retrieves the result of the embedding evaluation.

        Returns:
            Any: A dictionary containing the results of all evaluated metrics.

        Raises:
            ValueError: If the result is not available.
        """
        if not self.result:
            raise ValueError("No result available. Have you run the evaluation?")
        return self.result

    def _verify_inferred_trajectory(self, inferred_embedding: Any) -> np.ndarray:
        """Verifies the inferred embedding.

        Ensures that the inferred embedding is a 2D numpy array.

        Args:
            inferred_embedding (Any): The inferred embedding.

        Returns:
            np.ndarray: The verified inferred embedding.

        Raises:
            ValueError: If the inferred embedding is not a 2D numpy array.
        """
        if not isinstance(inferred_embedding, np.ndarray):
            raise ValueError("Inferred embedding must be a numpy array.")
        if inferred_embedding.ndim != 2:
            raise ValueError("Inferred embedding must be a 2D numpy array.")
        return inferred_embedding

    def _prepare_before_subset(self, given_trajectory: Any, inferred_embedding: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the data before subsetting.

        Computes reference distances from the given trajectory graph using Isomap for comparison.

        Args:
            given_trajectory (Any): The given trajectory (networkx.MultiDiGraph).
            inferred_embedding (Any): The inferred embedding.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The prepared reference distances and inferred embedding.

        Raises:
            ValueError: Unknown embedding method is specified in `prepare_params_before_subset`.
        """
        self.logger.debug("Computing reference distances from given trajectory.")

        # Convert the trajectory graph to a distance matrix using shortest path lengths
        node_labels = list(given_trajectory.nodes)
        num_nodes = len(node_labels)
        node_indices = {label: idx for idx, label in enumerate(node_labels)}
        reference_distances = np.full((num_nodes, num_nodes), np.inf)

        # Compute shortest path lengths
        lengths = dict(nx.all_pairs_shortest_path_length(given_trajectory))
        for i, u in enumerate(node_labels):
            for v, length in lengths[u].items():
                j = node_indices[v]
                reference_distances[i, j] = length

        # Replace infinities with a large number (if any disconnected nodes)
        reference_distances[reference_distances == np.inf] = num_nodes

        # Ensure symmetry and zero diagonal
        reference_distances = np.minimum(reference_distances, reference_distances.T)
        np.fill_diagonal(reference_distances, 0)

        # Compute reference embedding using Isomap for comparison
        embedding_method = self.prepare_params_before_subset.get("method")
        if embedding_method == "isomap":
            n_components = self.prepare_params_before_subset.get("n_components", inferred_embedding.shape[1])
            isomap = Isomap(n_components=n_components, metric="precomputed")
            reference_embedding = isomap.fit_transform(reference_distances)
            self.logger.debug("Reference embedding computed using Isomap.")
        elif embedding_method == "none":
            reference_embedding = reference_distances
            self.logger.debug("Reference embedding set as distance matrix (no embedding).")
        else:
            raise ValueError(
                f"Unknown embedding method {embedding_method!r} specified in prepare_params_before_subset."
            )

        return reference_embedding, inferred_embedding

    def _subset(self, reference_embedding: Any, inferred_embedding: Any) -> Tuple[Any, Any]:
        """Subsets the data based on `subset_params`.

        For embeddings, subsetting involves selecting specific samples based on indices or labels.

        Args:
            reference_embedding (Any): The prepared reference embedding before subsetting.
            inferred_embedding (Any): The prepared inferred embedding before subsetting.

        Returns:
            Tuple[Any, Any]: The subsetted reference embedding and inferred embedding.
        """
        if not self.subset_params:
            self.logger.debug("No subset parameters provided. Returning original embeddings.")
            return reference_embedding, inferred_embedding

        indices = np.arange(inferred_embedding.shape[0])

        # Subsetting by indices specified in subset_params
        subset_indices = self.subset_params.get("indices", None)
        if subset_indices is not None:
            indices = np.array(subset_indices)
            self.logger.debug(f"Subsetting data to indices: {indices}")
        else:
            self.logger.debug("No valid subset indices provided. Using all data.")

        subset_reference = reference_embedding[indices]
        subset_inferred = inferred_embedding[indices]

        return subset_reference, subset_inferred

    def _prepare_after_subset(self, reference_embedding: Any, inferred_embedding: Any) -> Tuple[Any, Any]:
        """Prepares the data after subsetting.

        Optionally aligns the embeddings or normalizes them.

        Args:
            reference_embedding (Any): The reference embedding.
            inferred_embedding (Any): The inferred embedding.

        Returns:
            Tuple[Any, Any]: The prepared reference embedding and inferred embedding.
        """
        self.logger.debug("Preparing embeddings after subsetting.")

        if self.prepare_params_after_subset.get("align_embeddings", True):
            self.logger.debug("Aligning inferred embedding to reference embedding using Procrustes analysis.")
            # Perform Procrustes analysis to align inferred embedding to reference
            mtx1, mtx2, disparity = procrustes(reference_embedding, inferred_embedding)
            self.logger.debug(f"Procrustes analysis completed with disparity: {disparity}")
            return mtx1, mtx2
        else:
            self.logger.debug("Alignment of embeddings skipped.")
            return reference_embedding, inferred_embedding
