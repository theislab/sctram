#!/usr/bin/env python3

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import kendalltau, ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, mutual_info_score, r2_score
from sklearn.metrics.pairwise import cosine_similarity

from sctram.evaluate._metricsmixin._spatialmetricsmixin import SpatialMetricsMixin


class EmbeddingMetricsMixin(SpatialMetricsMixin):

    available_metrics = [
        "procrustes",
        "pearson_correlation",
        "spearman_correlation",
        "kendall_correlation",
        "mean_squared_error",
        "mean_absolute_error",
        "r2_score",
        "cosine_similarity",
        "embedding_distance",
        "alignment_score",
        "morans_i",
        "gearys_c",
        "local_morans_i",
        "getis_ord_gi_star",
        "wasserstein_distance",
        "ks_statistic",
        "mutual_information",
    ]

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
            elif metric == "r2_score":
                self._calculate_r2_score()
            elif metric == "cosine_similarity":
                self._calculate_cosine_similarity()
            elif metric == "embedding_distance":
                self._calculate_embedding_distance()
            elif metric == "alignment_score":
                self._calculate_alignment_score()
            elif metric == "morans_i":
                self._calculate_morans_i()
            elif metric == "gearys_c":
                self._calculate_gearys_c()
            elif metric == "local_morans_i":
                self._calculate_local_morans_i()
            elif metric == "getis_ord_gi_star":
                self._calculate_getis_ord_gi_star()
            elif metric == "wasserstein_distance":
                self._calculate_wasserstein_distance()
            elif metric == "ks_statistic":
                self._calculate_ks_statistic()
            elif metric == "mutual_information":
                self._calculate_mutual_information()
            else:
                self.logger.warning(f"Unknown metric {metric!r} specified. Skipping.")

    def _calculate_procrustes(self):
        """Calculates the Procrustes distance between the reference and inferred embeddings."""
        _, _, disparity = procrustes(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["procrustes"] = disparity
        self.logger.debug(f"Procrustes disparity: {disparity}")

    def _calculate_embedding_correlation(self, method: str = "pearson"):
        """Calculates the correlation between the reference and inferred embeddings.

        Args:
            method (str): The correlation method to use ('pearson', 'spearman', 'kendall').

        Raises:
            ValueError: When the method is not among allowed ones, 'pearson', 'spearman', 'kendall'.
        """
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
        """Calculates the Mean Squared Error between the reference and inferred embeddings."""
        mse = mean_squared_error(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["mean_squared_error"] = mse
        self.logger.debug(f"Mean Squared Error: {mse}")

    def _calculate_mean_absolute_error(self):
        """Calculates the Mean Absolute Error between the reference and inferred embeddings."""
        mae = mean_absolute_error(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["mean_absolute_error"] = mae
        self.logger.debug(f"Mean Absolute Error: {mae}")

    def _calculate_r2_score(self):
        """Calculates the R-squared (coefficient of determination) between the reference and inferred embeddings."""
        r2 = r2_score(self.prepared_after_subset_given, self.prepared_after_subset_inferred)
        self.result["r2_score"] = r2
        self.logger.debug(f"R-squared: {r2}")

    def _calculate_cosine_similarity(self):
        """Calculates the cosine similarity between the reference and inferred embeddings."""
        embedding1 = self.prepared_after_subset_given.flatten().reshape(1, -1)
        embedding2 = self.prepared_after_subset_inferred.flatten().reshape(1, -1)
        cosine_sim = cosine_similarity(embedding1, embedding2)[0][0]
        self.result["cosine_similarity"] = cosine_sim
        self.logger.debug(f"Cosine similarity: {cosine_sim}")

    def _calculate_embedding_distance(self):
        """Calculates the Euclidean distance between the reference and inferred embeddings."""
        distance = np.linalg.norm(self.prepared_after_subset_given - self.prepared_after_subset_inferred)
        self.result["embedding_distance"] = distance
        self.logger.debug(f"Euclidean distance between embeddings: {distance}")

    def _calculate_alignment_score(self):
        """Calculates an alignment score between the reference and inferred embeddings.

        Example: Average cosine similarity across components.
        """
        cosine_similarities = []
        for i in range(self.prepared_after_subset_given.shape[1]):
            component1 = self.prepared_after_subset_given[:, i]
            component2 = self.prepared_after_subset_inferred[:, i]
            cosine_sim = cosine_similarity(component1.reshape(1, -1), component2.reshape(1, -1))[0][0]
            cosine_similarities.append(cosine_sim)
        alignment_score = np.mean(cosine_similarities)
        self.result["alignment_score"] = alignment_score
        self.logger.debug(f"Alignment score (average cosine similarity across components): {alignment_score}")

    def _calculate_morans_i(self):
        """Calculates Moran's I for the reference embedding."""
        morans_i_values = []
        spatial_weights = self._compute_spatial_weights(
            self.prepared_after_subset_given, "embedding", **self.method_params
        )
        for dim in range(self.prepared_after_subset_given.shape[1]):
            x = self.prepared_after_subset_given[:, dim]
            morans_i = self.calculate_morans_i(x, spatial_weights)
            morans_i_values.append(morans_i)
            self.logger.debug(f"Moran's I for component {dim}: {morans_i}")
        average_morans_i = np.mean(morans_i_values)
        self.result["morans_i"] = average_morans_i
        self.logger.debug(f"Average Moran's I across components: {average_morans_i}")

    def _calculate_gearys_c(self):
        """Calculates Geary's C for the reference embedding."""
        gearys_c_values = []
        spatial_weights = self._compute_spatial_weights(
            self.prepared_after_subset_given, "embedding", **self.method_params
        )
        for dim in range(self.prepared_after_subset_given.shape[1]):
            x = self.prepared_after_subset_given[:, dim]
            gearys_c = self.calculate_gearys_c(x, spatial_weights)
            gearys_c_values.append(gearys_c)
            self.logger.debug(f"Geary's C for component {dim}: {gearys_c}")
        average_gearys_c = np.mean(gearys_c_values)
        self.result["gearys_c"] = average_gearys_c
        self.logger.debug(f"Average Geary's C across components: {average_gearys_c}")

    def _calculate_local_morans_i(self):
        """Calculates Local Moran's I (LISA) for the reference embedding."""
        lisa_values = []
        spatial_weights = self._compute_spatial_weights(
            self.prepared_after_subset_given, "embedding", **self.method_params
        )
        for dim in range(self.prepared_after_subset_given.shape[1]):
            x = self.prepared_after_subset_given[:, dim]
            lisa = self.calculate_lisa(x, spatial_weights)
            lisa_values.append(lisa)
            self.logger.debug(f"Local Moran's I for component {dim}: {lisa}")
        # Aggregate by averaging across components
        average_lisa = np.mean(lisa_values, axis=0)
        self.result["local_morans_i"] = average_lisa
        self.logger.debug(f"Average Local Moran's I across components: {average_lisa}")

    def _calculate_getis_ord_gi_star(self):
        """Calculates Getis-Ord Gi* statistic for the reference embedding."""
        gi_star_values = []
        spatial_weights = self._compute_spatial_weights(
            self.prepared_after_subset_given, "embedding", **self.method_params
        )
        for dim in range(self.prepared_after_subset_given.shape[1]):
            x = self.prepared_after_subset_given[:, dim]
            gi_star = self.calculate_getis_ord_gi_star(x, spatial_weights)
            gi_star_values.append(gi_star)
            self.logger.debug(f"Getis-Ord Gi* for component {dim}: {gi_star}")
        # Aggregate by averaging across components
        average_gi_star = np.mean(gi_star_values, axis=0)
        self.result["getis_ord_gi_star"] = average_gi_star
        self.logger.debug(f"Average Getis-Ord Gi* across components: {average_gi_star}")

    def _calculate_wasserstein_distance(self):
        """Calculates the Wasserstein distance between the reference and inferred embeddings."""
        wd = wasserstein_distance(
            self.prepared_after_subset_given.flatten(), self.prepared_after_subset_inferred.flatten()
        )
        self.result["wasserstein_distance"] = wd
        self.logger.debug(f"Wasserstein distance: {wd}")

    def _calculate_ks_statistic(self):
        """Calculates the Kolmogorov-Smirnov statistic between the reference and inferred embeddings."""
        statistic, p_value = ks_2samp(
            self.prepared_after_subset_given.flatten(), self.prepared_after_subset_inferred.flatten()
        )
        self.result["ks_statistic"] = statistic
        self.result["ks_p_value"] = p_value
        self.logger.debug(f"Kolmogorov-Smirnov statistic: {statistic}, p-value: {p_value}")

    def _calculate_mutual_information(self):
        """Calculates the Mutual Information between the reference and inferred embeddings."""
        # Discretize the embeddings
        bins = self.method_params.get("mi_bins", 10)
        inferred_discrete = np.digitize(
            self.prepared_after_subset_inferred.flatten(),
            bins=np.linspace(
                self.prepared_after_subset_inferred.min(), self.prepared_after_subset_inferred.max(), bins
            ),
        )
        reference_discrete = np.digitize(
            self.prepared_after_subset_given.flatten(),
            bins=np.linspace(self.prepared_after_subset_given.min(), self.prepared_after_subset_given.max(), bins),
        )
        mi = mutual_info_score(reference_discrete, inferred_discrete)
        self.result["mutual_information"] = mi
        self.logger.debug(f"Mutual Information: {mi}")
