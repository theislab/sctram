#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from sctram.evaluate._base import EvaluationBase


class DiffmapEvaluation(EvaluationBase):
    """Evaluation method to compare inferred diffusion map embeddings with a given trajectory.

    This class compares the inferred diffusion map embeddings (numpy array of shape [n_samples, n_components])
    with the reference diffusion map embeddings computed from the given trajectory.

    Multiple metrics are used to assess the similarity between the embeddings, including Procrustes distance,
    correlation between embeddings, mean squared error, etc.
    """

    available_metrics = [
        "procrustes",
        "pearson_correlation",
        "spearman_correlation",
        "kendall_correlation",
        "mean_squared_error",
        "mean_absolute_error",
        "cosine_similarity",
        "embedding_distance",
        "alignment_score",
    ]

    def __init__(
        self,
        method_params: Dict[str, Any],
        subset_params: Optional[Dict[str, Any]] = None,
        prepare_params_before_subset: Optional[Dict[str, Any]] = None,
        prepare_params_after_subset: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the Diffmap evaluation method."""
        super().__init__(
            method_params=method_params,
            subset_params=subset_params,
            prepare_params_before_subset=prepare_params_before_subset,
            prepare_params_after_subset=prepare_params_after_subset,
        )
        self.logger.debug(f"Initialized DiffmapEvaluation with metrics: {self.metrics}")

        if prepare_params_before_subset is None or "method" not in prepare_params_before_subset.keys():
            raise ValueError("Determine a method how to compute reference diffusion map from given trajectory.")

    def _verify_inferred_trajectory(self, inferred_diffmap: Any) -> np.ndarray:
        """Verifies the inferred diffusion map embeddings.

        Ensures that the inferred diffusion map embeddings are a 2D numpy array.

        Args:
            inferred_diffmap (Any): The inferred diffusion map embeddings.

        Returns:
            np.ndarray: The verified inferred diffusion map embeddings.

        Raises:
            ValueError: If the inferred diffusion map embeddings are not a 2D numpy array.
        """
        if not isinstance(inferred_diffmap, np.ndarray):
            raise ValueError("Inferred diffusion map embeddings must be a numpy array.")
        if inferred_diffmap.ndim != 2:
            raise ValueError("Inferred diffusion map embeddings must be a 2D array.")
        return inferred_diffmap

    def _prepare_before_subset(self, given_trajectory: Any, inferred_diffmap: Any) -> Tuple[Any, Any]:
        """Prepares the data before subsetting.

        Computes a reference diffusion map embedding from the given trajectory.

        Args:
            given_trajectory (Any): The given trajectory.
            inferred_diffmap (Any): The inferred diffusion map embeddings.

        Returns:
            Tuple[Any, Any]: The prepared reference diffusion map embeddings and inferred diffusion map embeddings.
        """
        self.logger.debug("Computing reference diffusion map embeddings from given trajectory.")

        method = self.prepare_params_before_subset.get("method")
        if method == "compute_diffmap":
            # Example: Compute diffusion map embeddings from given data
            n_components = self.prepare_params_before_subset.get("n_components", inferred_diffmap.shape[1])
            from sklearn.manifold import SpectralEmbedding

            embedding = SpectralEmbedding(n_components=n_components, affinity='precomputed')
            reference_diffmap = embedding.fit_transform(given_trajectory.adjacency_matrix)
            self.logger.debug("Reference diffusion map embeddings computed using SpectralEmbedding.")
        else:
            raise ValueError(f"Unknown method {method!r} for computing reference diffusion map.")

        return reference_diffmap, inferred_diffmap

    def _subset(self, reference_diffmap: Any, inferred_diffmap: Any) -> Tuple[Any, Any]:
        """Subsets the data based on `subset_params`.

        For embeddings, subsetting might involve selecting specific samples.

        Args:
            reference_diffmap (Any): The prepared reference diffusion map embeddings before subsetting.
            inferred_diffmap (Any): The prepared inferred diffusion map embeddings before subsetting.

        Returns:
            Tuple[Any, Any]: The subsetted reference and inferred diffusion map embeddings.
        """
        if not self.subset_params:
            self.logger.debug("No subset parameters provided. Returning original embeddings.")
            return reference_diffmap, inferred_diffmap

        indices = np.arange(reference_diffmap.shape[0])

        # Subsetting by labels or indices specified in subset_params
        subset_indices = self.subset_params.get("indices", None)
        if subset_indices is not None:
            indices = np.array(subset_indices)
            self.logger.debug(f"Subsetting data to indices: {indices}")
        else:
            self.logger.debug("No valid subset indices provided. Using all data.")

        subset_reference = reference_diffmap[indices]
        subset_inferred = inferred_diffmap[indices]

        return subset_reference, subset_inferred

    def _prepare_after_subset(self, reference_diffmap: Any, inferred_diffmap: Any) -> Tuple[Any, Any]:
        """Prepares the data after subsetting.

        Optionally aligns the embeddings or normalizes them.

        Args:
            reference_diffmap (Any): The reference diffusion map embeddings.
            inferred_diffmap (Any): The inferred diffusion map embeddings.

        Returns:
            Tuple[Any, Any]: The prepared reference and inferred diffusion map embeddings.
        """
        self.logger.debug("Preparing embeddings after subsetting.")

        if self.prepare_params_after_subset.get("align_embeddings", True):
            self.logger.debug("Aligning inferred embeddings to reference embeddings using Procrustes analysis.")
            # Align the inferred embeddings to the reference embeddings
            mtx1, mtx2, disparity = procrustes(reference_diffmap, inferred_diffmap)
            self.logger.debug(f"Procrustes analysis completed with disparity: {disparity}")
            return mtx1, mtx2
        else:
            self.logger.debug("Alignment of embeddings skipped.")
            return reference_diffmap, inferred_diffmap

    def _calculate(self):
        """Performs the evaluation by comparing the embeddings using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            if metric == "procrustes":
                self._calculate_procrustes()
            elif metric == "pearson_correlation":
                self._calculate_embedding_correlation(method="pearson")
            elif metric == "spearman_correlation":
                self._calculate_embedding_correlation(method="spearman")
            elif metric == "kendall_correlation":
                self._calculate_embedding_correlation(method="kendall")
            elif metric == "mean_squared_error":
                self._calculate_mean_squared_error()
            elif metric == "mean_absolute_error":
                self._calculate_mean_absolute_error()
            elif metric == "cosine_similarity":
                self._calculate_cosine_similarity()
            elif metric == "embedding_distance":
                self._calculate_embedding_distance()
            elif metric == "alignment_score":
                self._calculate_alignment_score()
            else:
                self.logger.warning(f"Unknown metric {metric!r} specified. Skipping.")

    def _calculate_procrustes(self):
        """Calculates the Procrustes distance between the reference and inferred embeddings."""
        _, _, disparity = procrustes(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["procrustes"] = disparity
        self.logger.debug(f"Procrustes disparity: {disparity}")

    def _calculate_embedding_correlation(self, method="pearson"):
        """Calculates the correlation between the flattened embeddings."""
        embedding1 = self.prepared_after_subset_given.flatten()
        embedding2 = self.prepared_after_subset_inferred.flatten()

        if method == "pearson":
            corr, _ = pearsonr(embedding1, embedding2)
            self.result["pearson_correlation"] = corr
            self.logger.debug(f"Pearson correlation: {corr}")
        elif method == "spearman":
            corr, _ = spearmanr(embedding1, embedding2)
            self.result["spearman_correlation"] = corr
            self.logger.debug(f"Spearman correlation: {corr}")
        elif method == "kendall":
            tau, _ = kendalltau(embedding1, embedding2)
            self.result["kendall_correlation"] = tau
            self.logger.debug(f"Kendall's tau correlation: {tau}")
        else:
            raise ValueError(f"Unknown correlation method {method!r}.")

    def _calculate_mean_squared_error(self):
        """Calculates the Mean Squared Error between the embeddings."""
        mse = mean_squared_error(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["mean_squared_error"] = mse
        self.logger.debug(f"Mean Squared Error: {mse}")

    def _calculate_mean_absolute_error(self):
        """Calculates the Mean Absolute Error between the embeddings."""
        mae = mean_absolute_error(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["mean_absolute_error"] = mae
        self.logger.debug(f"Mean Absolute Error: {mae}")

    def _calculate_cosine_similarity(self):
        """Calculates the cosine similarity between the embeddings."""
        embedding1 = self.prepared_after_subset_given.flatten().reshape(1, -1)
        embedding2 = self.prepared_after_subset_inferred.flatten().reshape(1, -1)
        cosine_sim = cosine_similarity(embedding1, embedding2)[0][0]
        self.result["cosine_similarity"] = cosine_sim
        self.logger.debug(f"Cosine similarity: {cosine_sim}")

    def _calculate_embedding_distance(self):
        """Calculates the Euclidean distance between the embeddings."""
        distance = np.linalg.norm(self.prepared_after_subset_given - self.prepared_after_subset_inferred)
        self.result["embedding_distance"] = distance
        self.logger.debug(f"Euclidean distance between embeddings: {distance}")

    def _calculate_alignment_score(self):
        """Calculates an alignment score between the embeddings."""
        # Example: average cosine similarity across components
        cosine_similarities = []
        for i in range(self.prepared_after_subset_given.shape[1]):
            component1 = self.prepared_after_subset_given[:, i]
            component2 = self.prepared_after_subset_inferred[:, i]
            cosine_sim = cosine_similarity(component1.reshape(1, -1), component2.reshape(1, -1))[0][0]
            cosine_similarities.append(cosine_sim)
        alignment_score = np.mean(cosine_similarities)
        self.result["alignment_score"] = alignment_score
        self.logger.debug(f"Alignment score (average cosine similarity across components): {alignment_score}")

    def get_result(self) -> Any:
        """Retrieves the result of the trajectory evaluation.

        Returns:
            Any: A dictionary containing the results of all evaluated metrics.

        Raises:
            ValueError: If the result is not available.
        """
        if not self.result:
            raise ValueError("No result available. Have you run the evaluation?")
        return self.result