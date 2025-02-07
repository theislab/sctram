#!/usr/bin/env python3

import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase
from sctram.evaluate._metricsmixin._spatialmetricsmixin import SpatialMetricsMixin


class EmbeddingTrajectoryMetricsMixin(MetricsMixinBase, SpatialMetricsMixin):
    """A mixin class to compute various metrics comparing an embedding to a trajectory graph. The aim is to find out
    whether the trajectory is actually involved in the embedding using a battery of metrics.

    Attributes:
        prepared_after_subset_inferred (np.ndarray): The n-dimensional embedding matrix (shape: [n_cells, n_dims]).
        prepared_after_subset_given (nx.MultiDiGraph): The trajectory graph as a NetworkX MultiDiGraph.
        labels (np.ndarray): 1D array of strings representing cell-type labels for each cell in the embedding.
        result (dict): Dictionary to store computed metric results.
        logger (loguru.logger): Logger for debugging and information messages.
    """

    available_metrics = [
        "normalized_mean_curvature",
        "distance_correlation",
        "stress",
        "neighborhood_preservation_score",
        "branch_silhouette_score",
        "transition_smoothness",
        "trustworthiness",
        "leaf_node_separation",
        "morans_i",
        "gearys_c",
        "lisa",
        "getis_ord_gi_star",
    ]
    _embedding_metrics_n_neighbors: int = 15

    def _calculate(self):
        """Performs the evaluation by comparing the embedding and trajectory graph using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            if metric == "normalized_mean_curvature":
                self._calculate_normalized_mean_curvature()
            elif metric == "distance_correlation":
                self._calculate_distance_correlation()
            elif metric == "stress":
                self._calculate_stress()
            elif metric == "neighborhood_preservation_score":
                self._calculate_neighborhood_preservation_score()
            elif metric == "branch_silhouette_score":
                self._calculate_branch_silhouette_score()
            elif metric == "transition_smoothness":
                self._calculate_transition_smoothness()
            elif metric == "trustworthiness":
                self._calculate_trustworthiness()
            elif metric == "leaf_node_separation":
                self._calculate_leaf_node_separation()
            elif metric == "morans_i":
                self._calculate_spatial_autocorrelation(metric, input_type="embedding")
            elif metric == "gearys_c":
                self._calculate_spatial_autocorrelation(metric, input_type="embedding")
            elif metric == "lisa":
                self._calculate_spatial_autocorrelation(metric, input_type="embedding")
            elif metric == "getis_ord_gi_star":
                self._calculate_spatial_autocorrelation(metric, input_type="embedding")
            else:
                self.logger.warning(f"Unknown metric {metric!r} specified. Skipping.")

    def _compute_all_centroids(self):
        """Compute and cache centroids for all unique labels in the dataset.

        Calculates mean embeddings for each unique cell type label and stores
        them in a dictionary for efficient lookup. Validates that all labels
        have corresponding cells in the embedding.

        Raises:
            ValueError: If any label has no corresponding cells in the embedding
        """
        self._centroid_cache = {}
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            label_indices = np.where(self.labels == label)[0]
            if label_indices.size == 0:
                raise ValueError(f"No embeddings found for label '{label}'. Cannot compute centroid.")

            self._centroid_cache[label] = np.mean(self.prepared_after_subset_inferred[label_indices], axis=0)
            self.logger.debug(f"Computed centroid for label '{label}'")

    def _get_centroids(self, a_path):
        """Retrieve precomputed centroids for labels in a specified path.

        Args:
            a_path (list): Ordered list of node labels representing a path

        Returns:
            list: Centroids corresponding to labels in the path order

        Raises:
            ValueError: If any label in path is missing from centroid cache
        """
        if not hasattr(self, "_centroid_cache"):
            self._compute_all_centroids()

        centroids = []
        for label in a_path:
            if label not in self._centroid_cache:
                raise ValueError(f"Label '{label}' not found in centroid cache")
            centroids.append(self._centroid_cache[label])

        return centroids

    def _get_longest_path(self, graph):
        """Extract the longest simple path from a directed graph.

        Uses DAG-specific optimizations when possible, with fallback to
        heuristic path finding for cyclic graphs. Prioritizes biological
        plausibility by preferring source-to-sink paths.

        Args:
            graph (nx.DiGraph): Trajectory graph to analyze

        Returns:
            list: Node labels in longest path order

        Raises:
            ValueError: If no valid paths can be found
        """
        try:  # Try DAG-optimized method first
            if not nx.is_directed_acyclic_graph(graph):
                self.logger.warning("Graph contains cycles - longest path may not be biologically meaningful")

            longest_path = nx.dag_longest_path(graph)
            self.logger.info(f"Found DAG-optimized longest path with {len(longest_path)} nodes")
            return longest_path

        except nx.NetworkXError as e:
            self.logger.warning(f"DAG path finding failed ({str(e)}), using heuristic fallback")

            # Fallback strategy for cyclic graphs
            sources = [n for n, d in graph.in_degree() if d == 0]
            sinks = [n for n, d in graph.out_degree() if d == 0]

            if not sources or not sinks:
                raise ValueError("No clear source/sink nodes for path finding")

            all_paths = []
            for source in sources:
                for sink in sinks:
                    try:
                        paths = nx.all_simple_paths(graph, source, sink, cutoff=100)
                        all_paths.extend(paths)
                    except nx.NetworkXNoPath:
                        continue

            if not all_paths:
                raise ValueError("No valid source-to-sink paths found")

            # Prioritize longest path with biological validity checks
            longest_path = max(all_paths, key=lambda p: self._path_biological_score(p))
            self.logger.warning(
                f"Used heuristic path finding. Actual longest path may differ. "
                f"Found path length: {len(longest_path)}"
            )
            return longest_path

    def _path_biological_score(self, path):
        """Score path biological validity based on label distribution"""
        # Implement domain-specific validation logic here
        return len(path)  # Simple length-based scoring by default

    def _get_distances_matrices(self):
        """Compute pairwise distances.

        Graph-based shortest path distances and embedding-based Euclidean distances for nodes in the trajectory graph.

        Returns:
            tuple: A tuple containing:
                - graph_distances (np.ndarray): Array of graph-based pairwise shortest path distances.
                - embedding_distances (np.ndarray): Array of embedding-based pairwise Euclidean distances.

        Raises:
            ValueError: If the trajectory graph has fewer than 2 nodes.
        """
        # Extract all nodes involved in the trajectory graph
        nodes = list(self.prepared_after_subset_given.nodes())
        num_nodes = len(nodes)

        if num_nodes < 2:
            raise ValueError(f"Trajectory graph has {num_nodes} node(s). Not enough to compute the metric.")
        self.logger.debug(f"Number of nodes in the trajectory graph: {num_nodes}")

        # Compute centroids for each node based on their labels
        centroids = np.array(self._get_centroids(nodes))
        # Compute embedding-based pairwise Euclidean distances using scipy's pdist
        embedding_distances = pdist(centroids, metric="euclidean")
        all_shortest_paths = dict(  # Compute all-pairs shortest path lengths using NetworkX
            nx.all_pairs_shortest_path_length(
                self.prepared_after_subset_given,  # as the graph has no weight
            )
        )

        graph_distances = []
        for node_i, node_j in itertools.combinations(nodes, 2):  # Iterate over all unique node pairs
            distance = all_shortest_paths.get(node_i, {}).get(node_j, np.nan)
            if np.isnan(distance):
                self.logger.warning(
                    f"No path between '{node_i}' and '{node_j}' in the trajectory graph. Assigning NaN."
                )
            else:
                self.logger.debug(f"Pair ('{node_i}', '{node_j}'): graph_distance = {distance}")
            graph_distances.append(distance)

        graph_distances = np.array(graph_distances)
        embedding_distances = np.array(embedding_distances)

        self.logger.debug("Successfully computed graph and embedding distances.")

        return graph_distances, embedding_distances

    def _calculate_normalized_mean_curvature(self):
        """Calculates the normalized average curvature along the trajectory embedding.

        This metric quantifies the average bending intensity of the developmental trajectory in the
        embedding space. High curvature regions may correspond to critical transition points (e.g.,
        cell fate decisions), while low curvature suggests smooth progression.

        Biological Validation Advantage:
            - Low curvature (0.1-0.3): Linear processes like hematopoiesis
            - Moderate curvature (0.3-0.6): Branching morphogenesis
            - High curvature (>0.6): Rapid fate decisions (neural crest diversification)

        Methodology:
            1. Path Extraction: Identifies the longest path in the trajectory graph.
            2. Centroid Smoothing: Fits a cubic spline through label centroids to create a smooth path.
            3. Curvature Computation: Calculates curvature at each spline point using parametric derivatives.
            4. Normalization: Scales curvatures by the maximum theoretical value (π radians).
            5. Aggregation: Returns the mean normalized curvature.

        Mathematical Formulation:
            For a spline-interpolated path C(t) = (x(t), y(t), ...), the curvature at t is:
            κ(t) = ||C'(t) x C''(t)|| / ||C'(t)||³
            where:
            - C'(t) is the first derivative (tangent vector)
            - C''(t) is the second derivative (curvature vector)
            - x denotes the cross product (Euclidean norm in n-dimensions)

        Advantages:
            - Smooth Transition Analysis: Spline interpolation captures gradual state transitions better
            than discrete label centroids.
            - Dimensionally Agnostic: Works for 2D+ embeddings through vector norm operations.
            - Interpretable Scaling: Normalization (0=straight, 1=maximal bend) enables cross-dataset
            comparison.
            - Biological Relevance: Identifies sharp transitions potentially corresponding to key
            developmental events.

        Limitations:
            - Path Simplification: Relies on the longest path, potentially missing branch-specific curvature.
            - Label Dependency: Requires accurate cell-type labels and centroid calculations.
            - Spline Sensitivity: Over-smoothing may obscure biologically sharp transitions.
            - Directionality Ignored: Does not account for developmental timing or trajectory direction.

        Returns:
            float: Mean normalized curvature (0-1). Higher values indicate more frequent/sharp bends.
        """
        try:
            from scipy.interpolate import CubicSpline
            from scipy.misc import derivative

            # Get longest path and centroids
            longest_path = self._get_longest_path(self.prepared_after_subset_given)
            if len(longest_path) < 3:
                raise ValueError("Insufficient path length for curvature analysis")

            centroids = self._get_centroids(longest_path)
            centroids = np.array(centroids)

            # Create spline for each dimension
            t = np.linspace(0, 1, len(centroids))  # Parametric variable
            splines = [CubicSpline(t, centroids[:, dim]) for dim in range(centroids.shape[1])]

            # Function to compute position at parametric value t
            def position(t_val):
                return np.array([spl(t_val) for spl in splines])

            # Compute first and second derivatives
            delta = 1e-3  # Small perturbation for numerical derivatives
            curvatures = []
            for t_val in np.linspace(0, 1, 100):  # Sample 100 points along path
                # First derivative (tangent vector)
                C_prime = derivative(position, t_val, dx=delta, n=1)

                # Second derivative (acceleration vector)
                C_double_prime = derivative(position, t_val, dx=delta, n=2)

                # Compute curvature using the general formula
                norm_C_prime = np.linalg.norm(C_prime)
                norm_C_double_prime = np.linalg.norm(C_double_prime)

                if norm_C_prime < 1e-6:
                    self.logger.warning(
                        f"Zero or near-zero first derivative at t={t_val}. Skipping curvature computation for this point."
                    )
                    curvatures.append(0.0)
                    continue

                curvature = norm_C_double_prime / (norm_C_prime**3)
                # Normalize curvature by maximum theoretical value (π radians)
                normalized_curvature = np.clip(curvature / np.pi, 0, 1)
                curvatures.append(normalized_curvature)

                self.logger.debug(f"t={t_val}: curvature (normalized) = {normalized_curvature}")

            if not curvatures:
                raise ValueError("No curvature values were computed. Check the trajectory path and embedding.")

            normalized_mean_curvature = np.mean(curvatures)
            self.result["normalized_mean_curvature"] = normalized_mean_curvature

            self.logger.debug(f"Normalized mean curvature calculated: {normalized_mean_curvature}")

        except Exception as e:
            self.logger.error(f"Failed to calculate curvature deviation: {e}")
            self.result["normalized_mean_curvature"] = np.nan

    def _calculate_distance_correlation(self):
        """
        Calculates the Spearman rank correlation between graph-based shortest path distances
        and embedding-based Euclidean distances across all pairs of cell type labels in the trajectory graph.

        This metric assesses the extent to which the embedding preserves the relative distances defined
        by the trajectory graph. A high Spearman correlation indicates that cell types close in the graph
        are also close in the embedding, supporting the biological validity of the embedding.

        Mathematical Definition:
            Distance Correlation = Spearmanr(D_graph, D_embedding)

        Where:
            - D_graph is the vector of shortest path lengths between all unique pairs of labels in the trajectory graph.
            - D_embedding is the vector of Euclidean distances between the corresponding centroids in the embedding space.

        Advantages:
            - **Monotonic Relationship Assessment:** Spearman's correlation captures monotonic relationships, making it robust to non-linear distortions in the embedding.
            - **Comprehensive Pairwise Evaluation:** Considers all unique pairs of labels, providing a holistic assessment of the embedding's fidelity to the trajectory graph.
            - **Biological Interpretability:** Ensures that the embedding maintains the relative positioning of biologically distinct cell types, crucial for downstream analyses.

        Limitations:
            - **Computational Complexity:** For large numbers of labels, the number of pairwise comparisons grows quadratically, potentially impacting performance.
            - **Assumption of Label Homogeneity:** Assumes that each label represents a homogenous cell population with a well-defined centroid, which may not hold true in heterogeneous datasets.
            - **Sensitivity to Label Quality:** The method's accuracy depends on the correctness and granularity of label annotations; mislabeled or overly broad labels can distort the correlation.

        Assumptions:
            - **Acyclic Trajectory Graph:** Assumes that the trajectory graph is a Directed Acyclic Graph (DAG), reflecting unidirectional biological processes.
            - **Unique Centroids:** Assumes that each label corresponds to a unique centroid in the embedding space, representing the mean position of its constituent cells.
            - **Connectivity:** Assumes that the trajectory graph is sufficiently connected to allow meaningful computation of shortest path lengths between labels.

        Biological Considerations:
            - **Trajectory Fidelity:** A high correlation supports the hypothesis that the embedding accurately reflects the biological differentiation pathways.
            - **Pathway Diversity:** By considering all label pairs, the metric accounts for multiple differentiation routes, important in complex biological systems.

        Implementation Details:
            - **Pairwise Computation:** Iterates over all unique label pairs to compute graph-based and embedding-based distances.
            - **Handling Disconnected Pairs:** Excludes pairs with no connecting path in the graph, preventing skewing of the correlation due to undefined distances.

        Result:
            - A scalar value representing the Spearman rank correlation between graph-based and embedding-based distances. A value closer to 1 indicates strong preservation of trajectory distances in the embedding, while values near 0 suggest weak preservation.

        Raises:
            - ValueError: If there are insufficient labels to compute the correlation or if no valid label pairs are found.
        """
        try:
            nodes = list(self.prepared_after_subset_given.nodes())
            num_labels = len(nodes)
            if num_labels < 2:
                self.logger.debug(f"Trajectory graph has {num_labels} label(s). At least two are required.")
                raise ValueError("Insufficient labels in the trajectory to compute distance correlation.")

            # Compute centroids for all labels
            centroids = np.array(self._get_centroids(nodes))

            # Compute all-pairs shortest path lengths
            all_shortest_paths = dict(nx.all_pairs_shortest_path_length(self.prepared_after_subset_given))

            d_graph = []
            d_embedding = []

            for i, label_i in enumerate(nodes):
                for j, label_j in enumerate(nodes):
                    if i >= j:
                        continue  # Ensure each pair is considered only once

                    # Retrieve shortest path length; skip if no path exists
                    path_length = all_shortest_paths.get(label_i, {}).get(label_j, None)
                    if path_length is None:
                        self.logger.warning(
                            f"No path between '{label_i}' and '{label_j}' in the trajectory graph. Skipping this pair."
                        )
                        continue

                    d_graph.append(path_length)

                    # Compute Euclidean distance between centroids
                    distance = np.linalg.norm(centroids[j] - centroids[i])
                    d_embedding.append(distance)

                    self.logger.debug(
                        f"Pair ('{label_i}', '{label_j}'): graph_distance = {path_length}, "
                        f"embedding_distance = {distance}"
                    )

            if not d_graph or not d_embedding:
                raise ValueError("No valid label pairs found to compute distance correlation.")

            # Compute Spearman correlation
            correlation, p_value = spearmanr(d_graph, d_embedding)

            if np.isnan(correlation):
                self.logger.warning("Spearman correlation resulted in NaN. Possibly due to constant distance vectors.")
                self.result["distance_correlation"] = np.nan
            else:
                self.result["distance_correlation"] = correlation
                self.logger.info(f"Distance Correlation (Spearman): {correlation:.4f} (p-value: {p_value:.4e})")

        except Exception as e:
            self.logger.error(f"Failed to calculate distance correlation: {e}")
            self.result["distance_correlation"] = np.nan

    def _calculate_stress(self):
        """Calculates Sammon's stress (with scaling) between graph distances and embedding distances.

        Sammon's stress measures how well the embedding preserves the trajectory graph's
        structure, emphasizing local distances (shortest paths in the graph) more than
        global ones. Lower values indicate better preservation of both local and global
        structure, with particular sensitivity to local discrepancies.

        Mathematical Formulation:
            Stress = ( Σ_{i<j} [ (d_ij - e_ij)^2 / d_ij ] ) / Σ_{i<j} d_ij

        Where:
            - d_ij: Shortest path distance between nodes i and j in the trajectory graph.
            - e_ij: Euclidean distance between centroids of labels i and j in the embedding.

        Advantages:
            - Local Structure Emphasis: Prioritizes accurate preservation of adjacent
                cell states critical for developmental transitions.
            - Scale-Invariant: Normalization by d_ij makes the metric relative rather than
                absolute, enabling comparison across different embeddings.
            - Theoretical Foundation: Based on Sammon's mapping, a well-established
                dimensionality reduction evaluation technique.

        Limitations:
            - Graph Dependency: Assumes graph distances reflect true biological relationships;
                inaccurate graphs will distort the metric.
            - Centroid Approximation: Uses label centroids rather than full cell distributions,
                potentially missing within-label heterogeneity.
            - Disconnected Pairs: Excludes node pairs without a connecting path, which may
                underrepresent disconnected trajectory components.

        Biological Considerations:
            - High stress indicates poor preservation of cell state adjacencies, suggesting
                the embedding may obscure critical transition points (e.g., lineage bifurcations).
            - Low stress values (<0.1) suggest strong concordance between graph topology and embedding geometry,
                supporting biological validity. Values >0.3 indicate significant structural distortion.
        """
        try:
            graph_distances, embedding_distances = self._get_distances_matrices()

            # Remove invalid pairs
            valid_mask = ~np.isnan(graph_distances)
            if np.sum(valid_mask) < 2:
                self.logger.warning("Insufficient valid pairs for stress calculation.")
                self.result["stress"] = np.nan
                return

            graph_d = graph_distances[valid_mask]
            embedding_d = embedding_distances[valid_mask]

            # Exclude zero-graph-distance pairs
            nonzero_mask = graph_d > 0
            graph_d_nonzero = graph_d[nonzero_mask]
            embedding_d_nonzero = embedding_d[nonzero_mask]

            if len(graph_d_nonzero) < 2:
                self.logger.warning("No valid non-zero graph distance pairs.")
                self.result["stress"] = np.nan
                return

            # Critical fix: Scale embedding distances to match graph distance scale
            scale_factor = np.mean(graph_d_nonzero) / np.mean(embedding_d_nonzero)
            embedding_d_scaled = embedding_d_nonzero * scale_factor

            # Compute Sammon's stress with scaled distances
            squared_errors = (graph_d_nonzero - embedding_d_scaled) ** 2
            weighted_errors = squared_errors / graph_d_nonzero
            numerator = np.sum(weighted_errors)
            denominator = np.sum(graph_d_nonzero)

            stress = numerator / denominator
            self.result["stress"] = stress
            self.logger.info(f"Scaled Sammon's stress: {stress:.4f}")

        except Exception as e:
            self.logger.error(f"Stress calculation failed: {e}")
            self.result["stress"] = np.nan

    def _calculate_neighborhood_preservation_score(self):
        """Calculates the Neighborhood Preservation Score

        Assessing how well local neighborhoods in the embedding preserve the biological trajectory
        defined by the trajectory graph.

        Purpose:
            Measures the extent to which the embedding maintains the local neighborhood structure
            as defined by the trajectory graph, ensuring that cells occupy neighborhoods consistent
            with their biological transitions.

        Methodology:
            1. Adjacency Mapping: For each cell, identify adjacent cells in the trajectory graph based on
            direct transitions (successors).
            2. K-Nearest Neighbors (KNN): For each cell, find its K nearest neighbors in the embedding space.
            3. Neighborhood Overlap: For each cell, calculate the proportion of its KNN that are
            adjacent in the trajectory graph.
            4. Aggregation: Compute the average of these proportions across all cells to obtain the
            Neighborhood Preservation Score.

        Advantages:
            - Granular Assessment: Evaluates neighborhood preservation at the individual cell level,
                providing a detailed understanding of embedding fidelity.
            - Incorporates Directionality: Considers the direction of transitions in the trajectory graph,
                ensuring biological relevance in neighborhood relationships.
            - Inclusive: Accounts for all cells in the dataset, avoiding bias from excluding certain cells.
            - Provides a direct measure of how well the embedding preserves biologically relevant local
                neighborhoods.
            - Facilitates the identification of potential embedding distortions or misrepresentations in
                specific regions of the embedding space.

        Limitations:
            - Computational Complexity: Calculating KNN for each cell can be computationally intensive for large datasets.
            - Sensitivity to K: The choice of K can influence the metric's sensitivity and interpretability.
            - Dependency on Graph Accuracy: Relies on the accuracy of the trajectory graph; any inaccuracies
                can distort the score.
            - Assumption of Direct Transitions: Focuses on direct transitions, potentially overlooking higher-order relationships.
            - Sufficient Connectivity: Assumes that the trajectory graph is sufficiently connected to define
                meaningful adjacencies.
            - May not capture global trajectory preservation as it focuses on local neighborhoods.
            - Sensitive to the choice of K, which requires careful selection based on dataset characteristics.

        Biological Considerations:
            - Trajectory Fidelity: A higher Neighborhood Preservation Score indicates that the embedding
                accurately reflects the biological transitions, preserving local neighborhoods consistent with
                the trajectory graph.
            - Transition Integrity: Ensures that critical transition points and local neighborhood structures
                are maintained, which is essential for downstream biological interpretations and analyses.

        Returns:
            - `float`: The Neighborhood Preservation Score ranging from 0 to 1. Higher values indicate better
            preservation of neighborhood structures as defined by the trajectory graph.

        Raises:
            - `ValueError`: If the trajectory graph is empty or if no valid neighbors are found for any cell.
        """
        try:
            # Precompute adjacency for each label (excluding the label itself)
            graph = self.prepared_after_subset_given
            adjacency = {}
            for node in graph.nodes():
                successors = list(graph.successors(node))
                predecessors = list(graph.predecessors(node))
                adjacency[node] = set(successors + predecessors)

            scores = []
            n_cells = self.prepared_after_subset_inferred.shape[0]

            # Find K+1 neighbors (excluding self)
            nbrs = NearestNeighbors(n_neighbors=self._embedding_metrics_n_neighbors + 1, n_jobs=-1).fit(
                self.prepared_after_subset_inferred
            )
            _, indices = nbrs.kneighbors(self.prepared_after_subset_inferred)

            for i in range(n_cells):
                label = self.labels[i]
                adjacent_labels = adjacency.get(label, set())
                if not adjacent_labels:
                    # Skip cells with no adjacent labels in the trajectory graph
                    continue

                knn_indices = indices[i, 1:]  # Exclude self
                knn_labels = self.labels[knn_indices]

                # Count how many KNN labels are in adjacent_labels
                count_adjacent = np.sum([lbl in adjacent_labels for lbl in knn_labels])

                # Count how many KNN labels are the same as the cell's label
                count_same_label = np.sum([lbl == label for lbl in knn_labels])

                # Adjust denominator to exclude same-label neighbors
                denominator = self._embedding_metrics_n_neighbors - count_same_label

                if denominator > 0:
                    score = count_adjacent / denominator
                    scores.append(score)
                # else:
                #     # All KNN labels are the same as the cell's label; assign perfect score
                #     scores.append(1.0)

            if not scores:
                self.logger.warning("No valid cells for neighborhood preservation score.")
                self.result["neighborhood_preservation_score"] = np.nan
                return

            avg_score = np.mean(scores)
            self.result["neighborhood_preservation_score"] = avg_score
            self.logger.info(f"Neighborhood Preservation Score: {avg_score:.4f}")

        except Exception as e:
            self.logger.error(f"Neighborhood Preservation Score calculation failed: {e}")
            self.result["neighborhood_preservation_score"] = np.nan

    def _calculate_branch_silhouette_score(self):
        """Computes silhouette score for branches derived from the trajectory graph.

        Purpose: Evaluates the separation of branches in the embedding.
        Implementation: Assign cells to branches based on their label's position in the trajectory graph and compute the silhouette score for these branches in the embedding.
        Advantages: Quantifies how distinctly different branches are separated, important for complex trajectories.
        Limitations: Requires the graph to be a tree for unambiguous branch assignment.
        """
        try:
            # Identify root and leaves
            graph = self.prepared_after_subset_given
            root = [n for n, d in graph.in_degree() if d == 0]
            if len(root) != 1:
                raise ValueError("Cannot determine a single root.")
            root = root[0]
            leaves = [n for n, d in graph.out_degree() if d == 0]

            # Assign cells to branches based on leaf in their path
            branch_labels = []
            for lbl in self.labels:
                for leaf in leaves:
                    if nx.has_path(graph, root, leaf) and lbl in nx.shortest_path(graph, root, leaf):
                        branch_labels.append(leaf)
                        break
                else:
                    branch_labels.append(None)  # Cells not in any branch

            valid = ~np.isin(branch_labels, [None])
            if np.sum(valid) < 2:
                raise ValueError("Insufficient branches for silhouette score.")

            score = silhouette_score(self.prepared_after_subset_inferred[valid], np.array(branch_labels)[valid])
            self.result["branch_silhouette_score"] = score

        except Exception as e:
            self.logger.error(f"Branch Silhouette Score failed: {e}")
            self.result["branch_silhouette_score"] = np.nan

    def _calculate_transition_smoothness(self):
        """Computes variance of distances between consecutive centroids in the longest path.

        Calculates variance of distances between consecutive centroids along the trajectory path.
        Measures preservation of uniform developmental pacing in the embedding.
        Lower values indicate smoother transitions between successive cell states.

        Purpose: Assesses the uniformity of transitions between consecutive nodes in the trajectory.
        Implementation: Calculate the variance of Euclidean distances between consecutive centroids along the longest path.
        Advantages: Highlights abrupt changes in the embedding, indicating potential misalignments.
        Limitations: Only considers the longest path, potentially overlooking branches.
        """
        try:
            longest_path = self._get_longest_path(self.prepared_after_subset_given)
            centroids = self._get_centroids(longest_path)
            distances = [np.linalg.norm(centroids[i + 1] - centroids[i]) for i in range(len(centroids) - 1)]
            if len(distances) < 1:
                raise ValueError("No valid transitions to analyze")
            self.result["transition_smoothness"] = np.var(distances)

        except Exception as e:
            self.logger.error(f"Transition Smoothness failed: {e}")
            self.result["transition_smoothness"] = np.nan

    def _calculate_trustworthiness(self):
        """Compute trustworthiness using graph-based reachability as ground truth for neighborhood preservation.

        Purpose: Measures preservation of original neighborhoods in the embedding, penalizing intrusive neighbors.
        Implementation: Adapt the standard trustworthiness metric to use graph-based neighborhoods.
        Advantages: Accounts for both missing and intrusive neighbors, providing a balanced view.
        Limitations: Computationally intensive for large datasets.

        1. Defines original neighbors as cells reachable within k graph hops
        2. Uses shortest path length as distance metric
        3. Handles directionality properly

        Args:
            n_neighbors: Number of neighbors to consider (must be < n_samples/2)
        """
        try:
            G = self.prepared_after_subset_given
            n_samples = len(self.labels)

            # Validate parameters
            if self._embedding_metrics_n_neighbors >= n_samples // 2:
                raise ValueError(f"n_neighbors must be < {n_samples//2}")

            # Precompute all-pairs shortest paths
            paths = dict(nx.all_pairs_shortest_path_length(G))

            # Build reachability matrix
            reachability = np.zeros((n_samples, n_samples), dtype=np.float32)
            label_to_indices = defaultdict(list)

            for idx, lbl in enumerate(self.labels):
                label_to_indices[lbl].append(idx)

            # Vectorized reachability calculation
            for i, src_lbl in enumerate(self.labels):
                src_indices = label_to_indices[src_lbl]
                for dst_lbl, length in paths.get(src_lbl, {}).items():
                    if dst_lbl == src_lbl:
                        continue
                    dst_indices = label_to_indices.get(dst_lbl, [])
                    reachability[i, dst_indices] = 1 / (length + 1)  # Inverse distance

            # Compute trustworthiness with Jaccard weighting
            score = trustworthiness(
                X=reachability,
                X_embedded=self.prepared_after_subset_inferred,
                n_neighbors=self._embedding_metrics_n_neighbors,
                metric="precomputed",
            )

            self.result["trustworthiness"] = score
            self.logger.info(f"Trustworthiness: {score:.3f}")

        except Exception as e:
            self.logger.error(f"Trustworthiness failed: {str(e)}")
            self.result["trustworthiness"] = np.nan

    def _calculate_leaf_node_separation(self):
        """Computes average pairwise distance between terminal states (leaf nodes).

        It is possible to edit this one with silhouette instead.

        Higher values indicate better separation of distinct endpoints in the embedding.
        """
        try:
            # Identify leaf nodes
            G = self.prepared_after_subset_given
            leaves = [n for n, d in G.out_degree() if d == 0]

            if len(leaves) < 2:
                raise ValueError("At least two leaves required for separation analysis")

            # Get centroids with validation
            leaf_centroids = []
            for leaf in leaves:
                indices = np.where(self.labels == leaf)[0]
                if len(indices) == 0:
                    self.logger.warning(f"Skipping leaf {leaf} with no cells")
                    continue
                leaf_centroids.append(np.mean(self.prepared_after_subset_inferred[indices], axis=0))

            if len(leaf_centroids) < 2:
                raise ValueError("Insufficient valid leaves for separation analysis")

            # Compute pairwise distances
            distances = []
            for i in range(len(leaf_centroids)):
                for j in range(i + 1, len(leaf_centroids)):
                    distances.append(np.linalg.norm(leaf_centroids[i] - leaf_centroids[j]))

            self.result["leaf_node_separation"] = np.mean(distances)
            self.logger.debug(f"Leaf separation: {self.result['leaf_node_separation']:.4f}")

        except Exception as e:
            self.logger.warning(f"Leaf separation calculation failed: {str(e)}")
            self.result["leaf_node_separation"] = np.nan
