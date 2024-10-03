#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class SpacePopulator:
    """Generates a dataset by populating around centroids and along edges between centroids."""

    def __init__(
        self,
        centroids: np.ndarray,
        adjacency_matrix: np.ndarray,
        labels: np.ndarray,
        total_points: int,
        centroid_proportion: float = 0.3,
        cluster_distribution: str = "gaussian",
        cluster_distribution_z: float = 1.96,
        edge_distribution: str = "reverse_gaussian",
        edge_noise_distribution_z: float = 1.96,
        edge_distribution_params: dict = None,
        random_state: int = None,
    ):
        """
        Initializes the DataGenerator.

        Args:
            centroids (np.ndarray): Centroid coordinates.
            adjacency_matrix (np.ndarray): Adjacency matrix with edge weights.
            labels (np.ndarray): Labels for each centroid/node.
            total_points (int): Total number of data points to generate.
            centroid_proportion (float, optional): Proportion of points around centroids. Defaults to 0.3.
            cluster_distribution (str, optional): Distribution for cluster points. Defaults to 'gaussian'.
            cluster_distribution_z (float, optional): Desired z-score for dispersion around centroids. Defaults to 1.96.
            edge_distribution (str, optional): Distribution for edge points. Defaults to 'reverse_gaussian'.
            edge_noise_distribution_z (float, optional): Desired z-score for dispersion along edges. Defaults to 1.96.
            edge_distribution_params (dict, optional): Parameters for edge distribution. Defaults to None.
            random_state (int, optional): Seed for reproducibility. Defaults to None.
        """
        self.centroids = centroids
        self.adjacency_matrix = adjacency_matrix
        self.labels = labels
        self.total_points = total_points
        self.centroid_proportion = centroid_proportion
        self.cluster_distribution_z = cluster_distribution_z
        self.edge_noise_distribution_z = edge_noise_distribution_z
        self.random_state = random_state
        self.cluster_distribution = cluster_distribution
        self.edge_distribution = edge_distribution
        self.edge_distribution_params = edge_distribution_params if edge_distribution_params else {}

        self._validate_inputs()  # Validate inputs
        self._set_random_state()  # Set random state for reproducibility
        self.cluster_std, self.edge_std = self._calculate_cluster_and_edge_std()
        self.label_id_dict = {i: j for i, j in enumerate(self.labels)}

    def _validate_inputs(self):
        """Validates the input parameters."""
        if not (0 < self.centroid_proportion < 1):
            raise ValueError("centroid_proportion must be between 0 and 1.")
        if self.centroids.ndim != 2:
            raise ValueError("centroids must be a 2D array.")
        if self.adjacency_matrix.ndim != 2:
            raise ValueError("adjacency_matrix must be a 2D array.")
        if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be square.")
        if self.centroids.shape[0] != self.adjacency_matrix.shape[0]:
            raise ValueError("Number of centroids must match adjacency_matrix dimensions.")
        if self.labels.shape[0] != self.centroids.shape[0]:
            raise ValueError("Number of labels must match number of centroids.")
        if self.total_points <= 0:
            raise ValueError("total_points must be a positive integer.")
        if self.cluster_distribution_z is not None and self.cluster_distribution_z <= 0:
            raise ValueError("cluster_z_score must be positive.")
        if self.edge_noise_distribution_z is not None and self.edge_noise_distribution_z <= 0:
            raise ValueError("edge_z_score must be positive.")

        valid_distributions = ["gaussian", "uniform"]
        if self.cluster_distribution not in valid_distributions:
            raise ValueError(
                f"Unsupported cluster_distribution: {self.cluster_distribution}. Supported distributions: {valid_distributions}"
            )

        valid_edge_distributions = ["uniform", "beta", "cosine", "gaussian", "reverse_gaussian", "triangular"]
        if self.edge_distribution not in valid_edge_distributions:
            raise ValueError(
                f"Unsupported edge_distribution: {self.edge_distribution}. Supported distributions: {valid_edge_distributions}"
            )

    def _set_random_state(self):
        """Sets the random state for reproducibility."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _calculate_cluster_and_edge_std(self):
        """
        Calculates the standard deviations for generating points around centroids
        and along edges based on the desired z-scores and centroid dispersion.

        Returns:
            tuple: (cluster_std, edge_std)
        """
        # Calculate centroid dispersion (standard deviation of centroid distances from the mean)
        centroid_distances = np.linalg.norm(self.centroids - self.centroids.mean(axis=0), axis=1)
        centroid_dispersion = np.std(centroid_distances)

        # Use z-score to calculate standard deviations
        cluster_std = centroid_dispersion / self.cluster_distribution_z
        edge_std = centroid_dispersion / self.edge_noise_distribution_z

        # Ensure standard deviations are positive
        if cluster_std <= 0:
            raise ValueError("Calculated cluster_std is non-positive. Check centroids and cluster_z_score.")
        if edge_std <= 0:
            raise ValueError("Calculated edge_std is non-positive. Check centroids and edge_z_score.")

        return cluster_std, edge_std

    def generate_data(self):
        """
        Generates the dataset based on the centroids, adjacency matrix, and labels.

        Returns:
            tuple: (features np.ndarray, annotations pandas DataFrame)
        """
        num_centroids, dimensionality = self.centroids.shape

        # Calculate number of points
        num_points_centroids = int(self.total_points * self.centroid_proportion)
        num_points_edges = self.total_points - num_points_centroids

        # Generate points around centroids
        data_points_centroids = self._generate_points_around_centroids(num_points=num_points_centroids)

        # Generate points along edges
        data_points_edges = self._generate_points_between_centroids(num_points=num_points_edges)

        # Combine data points
        data_points = pd.concat([data_points_centroids, data_points_edges], ignore_index=True)

        # Ensure exact number of total points
        data_points = self._adjust_total_points(data_points)

        # Add centroid information
        centroid_information = pd.DataFrame(
            np.full(shape=(num_centroids, data_points.shape[1] - dimensionality), fill_value=None),
            columns=data_points.columns[dimensionality:],
        )
        centroid_information["type"] = "centroids"
        centroid_information["node_index"] = list(range(len(self.labels)))
        centroid_information["node_label"] = self.labels.copy()
        centroid_information["node_index_extended"] = centroid_information["node_index"]
        centroid_information["node_label_extended"] = centroid_information["node_label"].copy()
        centroids_df = pd.DataFrame(self.centroids, columns=[f"feature_{i}" for i in range(dimensionality)])
        centroids_df = pd.concat([centroids_df, centroid_information], axis=1)
        data_points = pd.concat([centroids_df, data_points], axis=0, ignore_index=True)

        features = data_points.iloc[:, :dimensionality].values
        annotations = data_points.iloc[:, dimensionality:]

        return features, annotations

    def _generate_points_around_centroids(self, num_points: int) -> pd.DataFrame:
        """
        Generates data points around each centroid using the specified distribution.

        Args:
            num_points (int): Total number of points to generate.

        Returns:
            pd.DataFrame: DataFrame containing generated data points and annotations.
        """
        num_centroids, dimensionality = self.centroids.shape
        points_per_centroid = num_points // num_centroids
        remainder = num_points % num_centroids
        data_frames = []

        for idx, (centroid, label) in enumerate(zip(self.centroids, self.labels)):
            n_points = points_per_centroid + (1 if idx < remainder else 0)
            if n_points <= 0:
                continue
            if self.cluster_distribution == "gaussian":
                points = np.random.normal(loc=centroid, scale=self.cluster_std, size=(n_points, dimensionality))
            elif self.cluster_distribution == "uniform":
                # The range is set to achieve the desired standard deviation
                range_width = self.cluster_std * 2 * np.sqrt(3)
                lower = centroid - range_width / 2
                upper = centroid + range_width / 2
                points = np.random.uniform(low=lower, high=upper, size=(n_points, dimensionality))
            else:
                raise ValueError(f"Unsupported cluster_distribution: {self.cluster_distribution}")
            df = pd.DataFrame(points, columns=[f"feature_{i}" for i in range(dimensionality)])
            df["type"] = "cluster"
            df["node_index"] = idx
            df["node_label"] = label
            df["edge_indices"] = np.nan
            df["edge_labels"] = np.nan
            df["node_index_extended"] = idx
            df["node_label_extended"] = label
            data_frames.append(df)

        if data_frames:
            data_points = pd.concat(data_frames, ignore_index=True)
        else:
            # No cluster points generated
            data_points = pd.DataFrame(
                columns=[f"feature_{i}" for i in range(dimensionality)]
                + [
                    "type",
                    "node_index",
                    "node_label",
                    "edge_indices",
                    "edge_labels",
                    "node_index_extended",
                    "node_label_extended",
                ]
            )

        return data_points

    def _generate_points_between_centroids(self, num_points: int) -> pd.DataFrame:
        """
        Generates data points along the edges between centroids, with density proportional to edge weights.

        Args:
            num_points (int): Total number of points to generate.

        Returns:
            pd.DataFrame: DataFrame containing generated data points and annotations.
        """
        num_centroids, dimensionality = self.centroids.shape

        # Extract edges and their weights
        edges = []
        for i in range(num_centroids):
            for j in range(i + 1, num_centroids):
                weight = self.adjacency_matrix[i, j]
                if weight > 0:
                    edges.append((i, j, weight))

        if not edges:
            # No edges to generate points between
            return pd.DataFrame(
                columns=[f"feature_{i}" for i in range(dimensionality)]
                + [
                    "type",
                    "node_index",
                    "node_label",
                    "edge_indices",
                    "edge_labels",
                    "node_index_extended",
                    "node_label_extended",
                ]
            )

        total_edge_weight = sum([edge[2] for edge in edges])

        # Edge case: if total_edge_weight is zero (all weights are zero)
        if total_edge_weight == 0:
            # Distribute points equally among edges
            points_per_edge = [num_points // len(edges)] * len(edges)
            remainder = num_points % len(edges)
            for idx in range(remainder):
                points_per_edge[idx] += 1
        else:
            # Calculate points per edge proportional to edge weights
            points_per_edge = []
            for edge in edges:
                edge_weight = edge[2]
                num_edge_points = max(1, int((edge_weight / total_edge_weight) * num_points))
                points_per_edge.append(num_edge_points)
            # Adjust total number of points to match num_points
            total_allocated = sum(points_per_edge)
            difference = num_points - total_allocated
            if difference > 0:
                # Distribute the remaining points
                for idx in range(difference):
                    points_per_edge[idx % len(points_per_edge)] += 1
            elif difference < 0:
                # Remove excess points
                for idx in range(-difference):
                    edge_idx = idx % len(points_per_edge)
                    if points_per_edge[edge_idx] > 1:
                        points_per_edge[edge_idx] -= 1
                    else:
                        # Can't have zero or negative points, find the next one
                        continue

        data_frames = []

        for edge, num_edge_points in zip(edges, points_per_edge):
            i, j, weight = edge
            centroid_i = self.centroids[i]
            centroid_j = self.centroids[j]
            label_i = self.labels[i]
            label_j = self.labels[j]
            if num_edge_points <= 0:
                continue  # Skip if no points to generate

            # Generate coefficients based on edge_distribution
            coefficients = self._generate_edge_coefficients(
                self.edge_distribution, self.edge_distribution_params, num_edge_points
            )

            # Linear interpolation
            points = centroid_i + coefficients * (centroid_j - centroid_i)

            # Add noise
            noise = np.random.normal(loc=0, scale=self.edge_std, size=(num_edge_points, dimensionality))
            points += noise

            df = pd.DataFrame(points, columns=[f"feature_{k}" for k in range(dimensionality)])
            df["type"] = "edge"
            df["node_index"] = np.nan
            df["node_label"] = np.nan
            df["edge_indices"] = f"{i}-{j}"
            df["edge_labels"] = f"{label_i}-{label_j}"
            # Compute node_index_extended
            distances = cdist(points, self.centroids)
            closest_centroids = np.argmin(distances, axis=1)
            df["node_index_extended"] = closest_centroids
            df["node_label_extended"] = [self.label_id_dict.get(i, "other") for i in closest_centroids]
            data_frames.append(df)

        if data_frames:
            data_points = pd.concat(data_frames, ignore_index=True)
        else:
            # No edge points generated
            data_points = pd.DataFrame(
                columns=[f"feature_{i}" for i in range(dimensionality)]
                + [
                    "type",
                    "node_index",
                    "node_label",
                    "edge_indices",
                    "edge_labels",
                    "node_index_extended",
                    "node_label_extended",
                ]
            )

        return data_points

    @staticmethod
    def _generate_edge_coefficients(
        edge_distribution: str, edge_distribution_params: dict, num_edge_points: int
    ) -> np.ndarray:
        """
        Generates interpolation coefficients for edge points based on the specified distribution.

        Args:
            num_edge_points (int): Number of points to generate.

        Returns:
            np.ndarray: Coefficients for interpolation.
        """
        if edge_distribution == "uniform" or num_edge_points < 3:
            coefficients = np.random.uniform(0, 1, size=(num_edge_points, 1))
        elif edge_distribution == "gaussian":
            # Normal distribution centered at 0.5
            mu = edge_distribution_params.get("mu", 0.5)
            sigma = edge_distribution_params.get("sigma", 0.15)
            coefficients = np.random.normal(loc=mu, scale=sigma, size=(num_edge_points, 1))
            coefficients = np.clip(coefficients, 0, 1)
        elif edge_distribution == "beta":
            # Default parameters for a smoother distribution
            a = edge_distribution_params.get("a", 2)
            b = edge_distribution_params.get("b", 2)
            coefficients = np.random.beta(a=a, b=b, size=(num_edge_points, 1))
        elif edge_distribution == "reverse_gaussian":
            # U-shaped distribution using 1 - Gaussian PDF
            sigma = edge_distribution_params.get("sigma", 0.15)
            x = np.linspace(0, 1, num_edge_points)
            gaussian_pdf = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
            probabilities = 1 - gaussian_pdf / gaussian_pdf.max()
            probabilities /= probabilities.sum()
            coefficients = np.random.choice(x, size=num_edge_points, p=probabilities)
            coefficients = coefficients.reshape(-1, 1)
        elif edge_distribution == "cosine":
            # U-shaped distribution using cosine function
            power = edge_distribution_params.get("power", 1)
            x = np.linspace(0, 1, num_edge_points)
            cosine_values = 0.5 * (1 + np.cos(np.pi * x))
            probabilities = cosine_values**power
            probabilities /= probabilities.sum()
            coefficients = np.random.choice(x, size=num_edge_points, p=probabilities)
            coefficients = coefficients.reshape(-1, 1)
        elif edge_distribution == "triangular":
            # U-shaped triangular distribution similar to reverse_gaussian
            # Combine two symmetric triangular distributions: one peaked at 0, another at 1
            # Introduce sharpness parameter (default to 0.5 for a balanced distribution)
            sharpness = edge_distribution_params.get("sharpness", 0.5)
            sharpness = np.clip(sharpness, 0.0, 1.0)  # Ensure sharpness is within (0, 1)

            num_half = num_edge_points // 2  # Split the number of edge points
            num_remainder = num_edge_points - num_half

            # Parameters for the first triangular distribution (peaked at 0)
            left1 = 0.0
            mode1 = 0.0
            right1 = sharpness  # Adjusted based on sharpness

            # Parameters for the second triangular distribution (peaked at 1)
            left2 = 1.0 - sharpness  # Adjusted based on sharpness
            mode2 = 1.0
            right2 = 1.0

            # Generate two sets of coefficients
            coeffs1 = np.random.triangular(left1, mode1, right1, size=num_half)
            coeffs2 = np.random.triangular(left2, mode2, right2, size=num_remainder)

            # Combine and shuffle
            coefficients = np.concatenate([coeffs1, coeffs2]).reshape(-1, 1)
            np.random.shuffle(coefficients)  # Shuffle to mix coefficients from both distributions
        else:
            raise ValueError(f"Unsupported edge_distribution: {edge_distribution}")

        return coefficients

    def _adjust_total_points(self, data_points: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the number of data points to exactly match total_points.

        Args:
            data_points (pd.DataFrame): Combined data points.

        Returns:
            pd.DataFrame: Adjusted data points.
        """
        num_generated_points = data_points.shape[0]
        if num_generated_points > self.total_points:
            # Randomly select total_points data points
            data_points = data_points.sample(n=self.total_points, random_state=self.random_state).reset_index(drop=True)
        elif num_generated_points < self.total_points:
            # Generate additional points around centroids
            num_missing = self.total_points - num_generated_points
            extra_data = self._generate_points_around_centroids(num_missing)
            data_points = pd.concat([data_points, extra_data], ignore_index=True)
        # Final check
        assert data_points.shape[0] == self.total_points, f"Expected {self.total_points}, got {data_points.shape[0]}"
        return data_points
