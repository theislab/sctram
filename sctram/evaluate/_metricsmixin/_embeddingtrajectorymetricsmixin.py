#!/usr/bin/env python3

# TODO: Codebase is not tested and/or runned.
# Note: The codebase here actually belongs to the previous version of the codebase.
# It was kept as reference.

from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff, pdist
from scipy.stats import pearsonr, spearmanr

from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase

# Optional dependencies
try:
    import gudhi as gd  # type: ignore
except ImportError:
    gd = None

try:
    from fastdtw import fastdtw  # type: ignore
except ImportError:
    fastdtw = None


class EmbeddingTrajectoryMetricsMixin(MetricsMixinBase):
    """A mixin class to compute various metrics comparing an embedding to a trajectory graph.

    Attributes:
        prepared_after_subset_given (np.ndarray): The n-dimensional embedding matrix (shape: [n_cells, n_dims]).
        prepared_after_subset_inferred (nx.MultiDiGraph): The trajectory graph as a NetworkX MultiDiGraph.
        labels (np.ndarray): 1D array of strings representing cell-type labels for each cell in the embedding.
        trajectory_nodes (list): Ordered list of nodes representing the trajectory in the graph.
        graph_positions (dict, optional): Dictionary mapping graph nodes to their spatial positions (np.ndarray).
        metrics (list): List of metric names to compute.
        result (dict): Dictionary to store computed metric results.
        logger (logging.Logger): Logger for debugging and information messages.
    """

    available_metrics = [
        "average_curvature",
        "trajectory_length_ratio",
        "spearman_distance_correlation",
        "stress",
        "average_deviation_from_ideal_path",
        "hausdorff_distance",
        "geodesic_distance_correlation",
        "persistent_homology_distance",
        "procrustes_disparity",
        "dtw_distance",
    ]

    def __init__(
        self,
        prepared_after_subset_given: np.ndarray,
        prepared_after_subset_inferred: nx.MultiDiGraph,
        labels: np.ndarray,
        trajectory_nodes: list,
        graph_positions: Optional[dict] = None,
    ):
        """Initializes the TrajectoryEmbeddingMetricsMixin.

        Args:
            prepared_after_subset_given (np.ndarray): The n-dimensional embedding matrix.
            prepared_after_subset_inferred (nx.MultiDiGraph): The trajectory graph.
            labels (np.ndarray): 1D array of cell-type labels for the embedding.
            trajectory_nodes (list): Ordered list of nodes representing the trajectory.
            graph_positions (dict, optional): Mapping from graph nodes to their spatial positions.
        """
        self.prepared_after_subset_given = prepared_after_subset_given
        self.prepared_after_subset_inferred = prepared_after_subset_inferred
        self.labels = labels
        self.trajectory_nodes = trajectory_nodes
        self.graph_positions = graph_positions  # Optional

        self.metrics = self.available_metrics.copy()
        self.result = {}

    def _calculate(self):
        """Performs the evaluation by comparing the embedding and trajectory graph using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            if metric == "average_curvature":
                self._calculate_average_curvature()
            elif metric == "trajectory_length_ratio":
                self._calculate_trajectory_length_ratio()
            elif metric == "spearman_distance_correlation":
                self._calculate_spearman_distance_correlation()
            elif metric == "stress":
                self._calculate_stress()
            elif metric == "average_deviation_from_ideal_path":
                self._calculate_average_deviation_from_ideal_path()
            elif metric == "hausdorff_distance":
                self._calculate_hausdorff_distance()
            elif metric == "geodesic_distance_correlation":
                self._calculate_geodesic_distance_correlation()
            elif metric == "persistent_homology_distance":
                self._calculate_persistent_homology_distance()
            elif metric == "procrustes_disparity":
                self._calculate_procrustes_disparity()
            elif metric == "dtw_distance":
                self._calculate_dtw_distance()
            else:
                self.logger.warning(f"Unknown metric {metric!r} specified. Skipping.")

    def _calculate_average_curvature(self):
        """Calculates the average curvature along the trajectory in the embedding.

        Computes the discrete curvature at each internal point of the trajectory
        using the positions of consecutive points and averages the curvature values.

        Advantages:
            - Captures the bending behavior of the trajectory.
            - Sensitive to changes in direction along the trajectory.

        Limitations:
            - Requires at least three points along the trajectory.
            - May be sensitive to noise in the embedding positions.

        Result:
            - A single scalar value representing the average curvature.
        """
        positions = [self.prepared_after_subset_given[node] for node in self.trajectory_nodes]
        positions = np.array(positions)
        curvatures = []
        for i in range(1, len(positions) - 1):
            p0 = positions[i - 1]
            p1 = positions[i]
            p2 = positions[i + 1]
            # Compute vectors
            v1 = p1 - p0
            v2 = p2 - p1
            # Compute the angle between the vectors
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                continue
            cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
            # Ensure numerical stability
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            # Curvature is the change in angle over the arc length
            arc_length = (norm_v1 + norm_v2) / 2
            curvature = angle / arc_length if arc_length != 0 else 0
            curvatures.append(curvature)
        if curvatures:
            average_curvature = np.mean(curvatures)
            self.result["average_curvature"] = average_curvature
            self.logger.debug(f"Average curvature along trajectory: {average_curvature}")
        else:
            self.result["average_curvature"] = np.nan
            self.logger.warning("Insufficient data to compute curvature.")

    def _calculate_trajectory_length_ratio(self):
        """Calculates and compares lengths of trajectory segments in the embedding and the graph.

        Computes the Euclidean lengths of each segment in the embedding and compares them
        to the lengths (if available) from the graph.

        Advantages:
            - Assesses the preservation of relative distances along the trajectory.
            - Sensitive to stretching or compressing of segments in the embedding.

        Limitations:
            - Requires segment lengths from the graph for direct comparison.
            - May be influenced by scaling differences.

        Result:
            - A scalar value representing the total length difference or ratio.
        """
        # Compute embedding lengths
        embedding_positions = [self.prepared_after_subset_given[node] for node in self.trajectory_nodes]
        embedding_positions = np.array(embedding_positions)
        embedding_lengths = np.linalg.norm(np.diff(embedding_positions, axis=0), axis=1)
        total_embedding_length = np.sum(embedding_lengths)

        # Compute graph lengths
        graph_lengths = []
        for i in range(len(self.trajectory_nodes) - 1):
            node_a = self.trajectory_nodes[i]
            node_b = self.trajectory_nodes[i + 1]
            if self.prepared_after_subset_inferred.has_edge(node_a, node_b):
                # If multiple edges exist, take the first one
                edge_data = self.prepared_after_subset_inferred.get_edge_data(node_a, node_b)
                # Assuming edge length is stored as 'length' attribute; default to 1 if not present
                if isinstance(edge_data, dict):
                    # Get the first edge's length
                    first_key = next(iter(edge_data))
                    length = edge_data[first_key].get("length", 1.0)
                else:
                    length = 1.0
                graph_lengths.append(length)
            else:
                # If no direct edge, use shortest path length
                try:
                    length = nx.shortest_path_length(
                        self.prepared_after_subset_inferred, source=node_a, target=node_b, weight="length"
                    )
                except nx.NetworkXNoPath:
                    length = np.nan  # Undefined
                graph_lengths.append(length)
        graph_lengths = np.array(graph_lengths)
        total_graph_length = np.nansum(graph_lengths)

        # Compute length ratio
        length_ratio = total_embedding_length / total_graph_length if total_graph_length != 0 else np.nan
        self.result["trajectory_length_ratio"] = length_ratio
        self.logger.debug(f"Total trajectory length ratio (embedding/graph): {length_ratio}")

    def _calculate_spearman_distance_correlation(self):
        """Calculates Spearman's rank correlation between graph and embedding distances.

        Computes pairwise distances among nodes in both the graph and the embedding,
        then calculates Spearman's rank correlation between these two sets of distances.

        Advantages:
            - Captures the monotonic relationship between graph and embedding distances.
            - Sensitive to the preservation of relative ordering of distances.

        Limitations:
            - Does not account for exact distances, only their ranks.
            - May be less sensitive to uniform scaling.

        Result:
            - A scalar value between -1 and 1 representing the Spearman correlation coefficient.
        """
        nodes = self.trajectory_nodes
        embedding_positions = np.array([self.prepared_after_subset_given[node] for node in nodes])
        # Compute pairwise distances in the embedding
        embedding_distances = pdist(embedding_positions, metric="euclidean")
        # Compute pairwise graph distances
        graph_distances = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]
                try:
                    distance = nx.shortest_path_length(
                        self.prepared_after_subset_inferred, source=node_i, target=node_j, weight="length"
                    )
                except nx.NetworkXNoPath:
                    distance = np.nan  # Undefined
                graph_distances.append(distance)
        graph_distances = np.array(graph_distances)
        # Remove pairs with undefined distances
        valid_mask = ~np.isnan(graph_distances)
        if np.sum(valid_mask) == 0:
            rho = np.nan
            self.logger.warning("No valid graph distances available for Spearman correlation.")
        else:
            rho, _ = spearmanr(graph_distances[valid_mask], embedding_distances[valid_mask])
        self.result["spearman_distance_correlation"] = rho
        self.logger.debug(f"Spearman's rank correlation of distances: {rho}")

    def _calculate_stress(self):
        """Calculates the stress function between graph distances and embedding distances.

        Computes the stress function as the normalized sum of squared differences between
        the pairwise distances in the graph and the embedding.

        Advantages:
            - Quantifies the overall distortion in the embedding.
            - Sensitive to both global and local distance preservation.

        Limitations:
            - Influenced by scaling; may need normalization.
            - Aggregates all pairwise discrepancies, potentially hiding local issues.

        Result:
            - A scalar value representing the stress (lower is better).
        """
        nodes = self.trajectory_nodes
        embedding_positions = np.array([self.prepared_after_subset_given[node] for node in nodes])
        embedding_distances = pdist(embedding_positions, metric="euclidean")
        # Compute graph distances
        graph_distances = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]
                try:
                    distance = nx.shortest_path_length(
                        self.prepared_after_subset_inferred, source=node_i, target=node_j, weight="length"
                    )
                except nx.NetworkXNoPath:
                    distance = np.nan  # Undefined
                graph_distances.append(distance)
        graph_distances = np.array(graph_distances)
        embedding_distances = np.array(embedding_distances)
        # Remove pairs with undefined distances
        valid_mask = ~np.isnan(graph_distances)
        if np.sum(valid_mask) == 0:
            stress = np.nan
            self.logger.warning("No valid graph distances available for stress calculation.")
        else:
            graph_distances = graph_distances[valid_mask]
            embedding_distances = embedding_distances[valid_mask]
            # Normalize distances
            graph_distances_norm = (
                graph_distances / np.max(graph_distances) if np.max(graph_distances) != 0 else graph_distances
            )
            embedding_distances_norm = (
                embedding_distances / np.max(embedding_distances)
                if np.max(embedding_distances) != 0
                else embedding_distances
            )
            # Compute stress
            stress_numerator = np.sum((embedding_distances_norm - graph_distances_norm) ** 2)
            stress_denominator = np.sum(graph_distances_norm**2)
            stress = np.sqrt(stress_numerator / stress_denominator) if stress_denominator != 0 else np.nan
            self.logger.debug(f"Stress function: {stress}")
        self.result["stress"] = stress

    def _calculate_average_deviation_from_ideal_path(self):
        """Calculates the average deviation from an ideal trajectory path in the embedding.

        Projects each point in the embedding onto the ideal trajectory path and computes
        the orthogonal distance, then averages these distances.

        Advantages:
            - Directly measures how well the embedding aligns with the expected trajectory.
            - Sensitive to deviations perpendicular to the trajectory path.

        Limitations:
            - Requires a well-defined ideal trajectory path.
            - May be less informative if the trajectory path is complex.

        Result:
            - A scalar value representing the average deviation.
        """
        positions = [self.prepared_after_subset_given[node] for node in self.trajectory_nodes]
        positions = np.array(positions)
        # Define the ideal path as a straight line from start to end
        start_point = positions[0]
        end_point = positions[-1]
        ideal_direction = end_point - start_point
        norm = np.linalg.norm(ideal_direction)
        if norm == 0:
            self.result["average_deviation_from_ideal_path"] = np.nan
            self.logger.warning("Start and end points are identical. Cannot define ideal path.")
            return
        ideal_direction /= norm
        deviations = []
        for pos in positions:
            vector = pos - start_point
            projection_length = np.dot(vector, ideal_direction)
            projection_point = start_point + projection_length * ideal_direction
            deviation = np.linalg.norm(pos - projection_point)
            deviations.append(deviation)
        average_deviation = np.mean(deviations)
        self.result["average_deviation_from_ideal_path"] = average_deviation
        self.logger.debug(f"Average deviation from ideal path: {average_deviation}")

    def _calculate_hausdorff_distance(self):
        """Calculates the Hausdorff distance between the graph trajectory and embedding trajectory.

        Treats the sequences of nodes in the graph and their corresponding positions in
        the embedding as point sets and computes the Hausdorff distance.

        Advantages:
            - Captures the worst-case deviation between the trajectories.
            - Sensitive to outliers or significant deviations.

        Limitations:
            - May be overly influenced by a single point.
            - Requires corresponding points between graph and embedding.

        Result:
            - A scalar value representing the Hausdorff distance.
        """
        if self.graph_positions is None:
            self.logger.warning("Graph positions are not provided. Hausdorff distance cannot be computed.")
            self.result["hausdorff_distance"] = np.nan
            return
        positions = [self.prepared_after_subset_given[node] for node in self.trajectory_nodes]
        positions = np.array(positions)
        # Retrieve graph positions in the same order
        try:
            graph_positions = np.array([self.graph_positions[node] for node in self.trajectory_nodes])
        except KeyError as e:
            self.logger.error(f"Graph position for node {e} not found. Hausdorff distance cannot be computed.")
            self.result["hausdorff_distance"] = np.nan
            return
        # Compute directed Hausdorff distances
        forward_hausdorff = directed_hausdorff(positions, graph_positions)[0]
        backward_hausdorff = directed_hausdorff(graph_positions, positions)[0]
        hausdorff_distance = max(forward_hausdorff, backward_hausdorff)
        self.result["hausdorff_distance"] = hausdorff_distance
        self.logger.debug(f"Hausdorff distance between trajectories: {hausdorff_distance}")

    def _calculate_geodesic_distance_correlation(self):
        """Calculates the correlation between graph geodesic distances and embedding distances.

        Computes the Pearson correlation coefficient between the shortest path lengths
        in the graph and the Euclidean distances in the embedding.

        Advantages:
            - Assesses the preservation of intrinsic geometric relationships.
            - Sensitive to both local and global distance distortions.

        Limitations:
            - May be influenced by scaling and outliers.
            - Assumes linear relationship between distances.

        Result:
            - A scalar value between -1 and 1 representing the Pearson correlation coefficient.
        """
        nodes = self.trajectory_nodes
        embedding_positions = np.array([self.prepared_after_subset_given[node] for node in nodes])
        embedding_distances = pdist(embedding_positions, metric="euclidean")
        # Compute graph distances
        graph_distances = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]
                try:
                    distance = nx.shortest_path_length(
                        self.prepared_after_subset_inferred, source=node_i, target=node_j, weight="length"
                    )
                except nx.NetworkXNoPath:
                    distance = np.nan  # Undefined
                graph_distances.append(distance)
        graph_distances = np.array(graph_distances)
        embedding_distances = np.array(embedding_distances)
        # Remove pairs with undefined distances
        valid_mask = ~np.isnan(graph_distances)
        if np.sum(valid_mask) == 0:
            correlation = np.nan
            self.logger.warning("No valid graph distances available for Pearson correlation.")
        else:
            correlation, _ = pearsonr(graph_distances[valid_mask], embedding_distances[valid_mask])
        self.result["geodesic_distance_correlation"] = correlation
        self.logger.debug(f"Geodesic distance preservation (Pearson correlation): {correlation}")

    def _calculate_persistent_homology_distance(self):
        """Calculates the similarity of persistent homology between the graph and embedding.

        Computes persistence diagrams for both the graph and the embedding and
        calculates the Wasserstein distance between them.

        Advantages:
            - Captures higher-order topological features.
            - Sensitive to both local and global topological similarities.

        Limitations:
            - Requires specialized libraries (e.g., Gudhi).
            - Interpretation of results may require expertise in TDA.

        Result:
            - A scalar value representing the Wasserstein distance between persistence diagrams.
        """
        if gd is None:
            self.logger.error("GUDHI is not installed. Persistent Homology Distance cannot be computed.")
            self.result["persistent_homology_distance"] = np.nan
            return
        # Compute persistence diagram for embedding
        embedding_positions = np.array([self.prepared_after_subset_given[node] for node in self.trajectory_nodes])
        rips_embedding = gd.RipsComplex(points=embedding_positions)
        simplex_tree_embedding = rips_embedding.create_simplex_tree(max_dimension=2)
        simplex_tree_embedding.compute_persistence()
        diag_embedding = simplex_tree_embedding.persistence_intervals_in_dimension(1)

        # Compute persistence diagram for graph
        # Extract graph edge lengths; assume 'length' attribute exists, else default to 1
        graph_edge_lengths = []
        for edge in self.prepared_after_subset_inferred.edges(data=True):
            length = edge[2].get("length", 1.0)
            graph_edge_lengths.append(length)
        # Assign positions to graph nodes if not provided
        if self.graph_positions is None:
            # Assign arbitrary positions, e.g., based on shortest paths
            graph_positions = {}
            for node in self.trajectory_nodes:
                try:
                    path = nx.shortest_path(
                        self.prepared_after_subset_inferred,
                        source=self.trajectory_nodes[0],
                        target=node,
                        weight="length",
                    )
                    pos = np.sum([self.prepared_after_subset_given[n] for n in path], axis=0) / len(path)
                except nx.NetworkXNoPath:
                    pos = self.prepared_after_subset_given[node]
                graph_positions[node] = pos
        else:
            graph_positions = self.graph_positions
        graph_positions_array = np.array([graph_positions[node] for node in self.trajectory_nodes])

        rips_graph = gd.RipsComplex(points=graph_positions_array)
        simplex_tree_graph = rips_graph.create_simplex_tree(max_dimension=2)
        simplex_tree_graph.compute_persistence()
        diag_graph = simplex_tree_graph.persistence_intervals_in_dimension(1)

        # Compute Wasserstein distance between persistence diagrams
        if len(diag_graph) == 0 or len(diag_embedding) == 0:
            self.result["persistent_homology_distance"] = np.nan
            self.logger.warning("One or both persistence diagrams are empty. Cannot compute Wasserstein distance.")
            return
        persistence_distance = gd.wasserstein_distance(diag_graph, diag_embedding)
        self.result["persistent_homology_distance"] = persistence_distance
        self.logger.debug(f"Persistent homology Wasserstein distance: {persistence_distance}")

    def _calculate_procrustes_disparity(self):
        """Calculates the Procrustes alignment score between the graph trajectory and embedding trajectory.

        Aligns the trajectories and computes the disparity, which measures the
        dissimilarity between the two configurations after scaling, translating, and rotating.

        Advantages:
            - Provides a direct geometric comparison.
            - Invariant to scaling, translation, and rotation.

        Limitations:
            - Assumes correspondence between points in both trajectories.
            - Disparity may be influenced by outliers.

        Result:
            - A scalar value representing the Procrustes disparity (lower is better).
        """
        if self.graph_positions is None:
            self.logger.warning("Graph positions are not provided. Procrustes disparity cannot be computed.")
            self.result["procrustes_disparity"] = np.nan
            return
        positions = np.array([self.prepared_after_subset_given[node] for node in self.trajectory_nodes])
        try:
            graph_positions = np.array([self.graph_positions[node] for node in self.trajectory_nodes])
        except KeyError as e:
            self.logger.error(f"Graph position for node {e} not found. Procrustes disparity cannot be computed.")
            self.result["procrustes_disparity"] = np.nan
            return
        # Perform Procrustes analysis
        mtx1, mtx2, disparity = procrustes(graph_positions, positions)
        self.result["procrustes_disparity"] = disparity
        self.logger.debug(f"Procrustes disparity between trajectories: {disparity}")

    def _calculate_dtw_distance(self):
        """Calculates the Dynamic Time Warping distance between the graph and embedding trajectories.

        Uses DTW to align the sequences of positions and computes the minimal total distance.

        Advantages:
            - Accounts for non-linear variations in progression along the trajectory.
            - Sensitive to both spatial and temporal deviations.

        Limitations:
            - Requires sequences to be ordered.
            - Computationally intensive for long sequences.

        Result:
            - A scalar value representing the DTW distance.
        """
        if fastdtw is None:
            self.logger.error("fastdtw is not installed. DTW distance cannot be computed.")
            self.result["dtw_distance"] = np.nan
            return
        nodes = self.trajectory_nodes
        positions = [self.prepared_after_subset_given[node] for node in nodes]
        if self.graph_positions is None:
            self.logger.warning(
                "Graph positions are not provided. Assigning embedding positions to graph positions for DTW."
            )
            graph_positions = positions  # Assign embedding positions as graph positions
        else:
            try:
                graph_positions = [self.graph_positions[node] for node in nodes]
            except KeyError as e:
                self.logger.error(f"Graph position for node {e} not found. DTW distance cannot be computed.")
                self.result["dtw_distance"] = np.nan
                return
        positions = np.array(positions)
        graph_positions = np.array(graph_positions)
        # Compute DTW distance for each dimension and sum
        if positions.shape[1] != graph_positions.shape[1]:
            self.logger.error(
                "Embedding and graph positions have different dimensions. DTW distance cannot be computed."
            )
            self.result["dtw_distance"] = np.nan
            return
        total_distance = 0.0
        for dim in range(positions.shape[1]):
            distance, _ = fastdtw(positions[:, dim], graph_positions[:, dim], dist="euclidean")
            total_distance += distance
        self.result["dtw_distance"] = total_distance
        self.logger.debug(f"Dynamic Time Warping distance: {total_distance}")
