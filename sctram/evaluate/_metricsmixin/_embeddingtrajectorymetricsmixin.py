#!/usr/bin/env python3

# TODO: Codebase is not tested and/or runned.

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase


class EmbeddingTrajectoryMetricsMixin(MetricsMixinBase):
    """A mixin class to compute various metrics comparing an embedding to a trajectory graph. The aim is to find out
    whether the trajectory is actually involved in the embedding using a battery of metrics.

    Attributes:
        prepared_after_subset_inferred (np.ndarray): The n-dimensional embedding matrix (shape: [n_cells, n_dims]).
        prepared_after_subset_given (nx.MultiDiGraph): The trajectory graph as a NetworkX MultiDiGraph.
        labels (np.ndarray): 1D array of strings representing cell-type labels for each cell in the embedding.
        result (dict): Dictionary to store computed metric results.
        logger (logging.Logger): Logger for debugging and information messages.
    """

    available_metrics = [
        "curvature_deviation",
        "distance_correlation",
        "stress",
        "geodesic_distance_correlation",
    ]

    def _calculate(self):
        """Performs the evaluation by comparing the embedding and trajectory graph using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            if metric == "curvature_deviation":
                self._calculate_curvature_deviation()
            elif metric == "distance_correlation":
                self._calculate_distance_correlation()
            elif metric == "stress":
                self._calculate_stress()
            elif metric == "geodesic_distance_correlation":
                self._calculate_geodesic_distance_correlation()
            else:
                self.logger.warning(f"Unknown metric {metric!r} specified. Skipping.")

    def _get_longest_path(self, a_path):
        # Extract the longest path from the trajectory graph
        # Assuming the trajectory is a directed acyclic graph (DAG)
        if not nx.is_directed_acyclic_graph(a_path):
            self.logger.warning("Trajectory graph is not a DAG. Attempting to extract a simple path.")

        # Attempt to find the longest path; fallback to any path if not a DAG
        try:
            longest_path = nx.dag_longest_path(a_path)  # Correct function usage
        except nx.NetworkXError as e:
            self.logger.error(f"Failed to compute the longest path due to an error: {e}")
            # If not a DAG, use a simple path (e.g., breadth-first search)
            paths = list(
                nx.all_simple_paths(
                    a_path,
                    source=list(a_path.nodes())[0],
                    target=list(a_path.nodes())[-1],
                    cutoff=1000  # Adjust cutoff as needed
                )
            )
            if not paths:
                raise ValueError("No simple paths found in the trajectory graph.")
            # Choose the longest among the found paths
            longest_path = max(paths, key=len)
        # TODO: add a warning if "longest_path != path"
        return longest_path

    def _get_centroids(self, a_path):
        # Compute centroids for each label in the longest path
        centroids = []
        for label in a_path:
            # Find indices of embeddings with the current label
            label_indices = np.where(self.labels == label)[0]

            if label_indices.size == 0:
                self.logger.error(f"No embeddings found for label '{label}'. Cannot compute centroid.")
                raise ValueError(f"No embeddings found for label '{label}'.")

            # Extract embeddings for the current label
            label_embeddings = self.prepared_after_subset_inferred[label_indices]

            # Compute centroid (mean across all embeddings of this label)
            centroid = np.mean(label_embeddings, axis=0)
            centroids.append(centroid)

            self.logger.debug(f"Label '{label}': centroid = {centroid}")
        return centroids

    def _calculate_curvature_deviation(self):
        """Calculates the standard deviation of curvature along the trajectory in the embedding.

        This method computes the discrete curvature at each internal point of the trajectory
        by analyzing the positions of consecutive points in the embedding. It then calculates
        the standard deviation of these curvature values to assess the variability of the trajectory's bending.

        Advantages:
            - Generalization: Uses centroids of cell-type labels for a more generalized curvature analysis along the
                trajectory, mitigating individual cell variability.
            - Sparse Data Handling: Effective even with sparse data for some labels, assuming
                enough points are available for centroid calculation.

        Limitations:
            - Label Dependence: Accuracy depends heavily on consistent and correct labeling.
            - Label Distribution: Sensitive to the distribution and representation of cells across labels.
            - Path Assumptions: Assumes the curvature in the latent space is similar to the physical or
                biological trajectory, which may not always hold.
            - Vector Norms: Sensitive to the norms of vectors between centroids; extreme values
                can skew curvature measurements.

        Raises:
            ValueError: If the trajectory graph does not contain enough nodes to compute curvature.
        """
        try:
            longest_path = self._get_longest_path(self.prepared_after_subset_given)
            num_points = len(longest_path)
            if num_points < 3:
                self.logger.debug(f"Longest trajectory path has {num_points} points.")
                raise ValueError("Not enough points in the trajectory to compute curvature.")

            centroids = self._get_centroids(longest_path)
            num_points = len(centroids)

            # Initialize list to store curvature values
            curvatures = []

            for i in range(1, num_points - 1):
                p_prev = centroids[i - 1]
                p_curr = centroids[i]
                p_next = centroids[i + 1]

                # Compute vectors
                v1 = p_curr - p_prev
                v2 = p_next - p_curr

                # Compute the norms
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)

                if norm_v1 == 0 or norm_v2 == 0:
                    self.logger.warning(
                        f"Zero-length segment at label index {i}. Skipping curvature computation for this point."
                    )
                    continue

                # Normalize vectors
                v1_normalized = v1 / norm_v1
                v2_normalized = v2 / norm_v2

                # Compute the cosine of the angle between vectors
                cos_theta = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
                angle = np.arccos(cos_theta)

                # Curvature can be defined as the angle change
                curvature = angle
                curvatures.append(curvature)

                self.logger.debug(f"Centroid {i}: angle (radians) = {curvature}")

            if not curvatures:
                raise ValueError("No curvature values were computed. Check the trajectory path and embedding.")

            curvature_std_dev = np.std(curvatures)
            curvature_mean = np.mean(curvatures)
            self.result["curvature_deviation"] = curvature_std_dev
            self.result["curvature_mean"] = curvature_mean

            self.logger.debug(f"Curvature deviation calculated: {curvature_std_dev}, mean {curvature_mean}")

        except Exception as e:
            self.logger.debug(f"Failed to calculate curvature deviation: {e}")
            self.result["curvature_deviation"] = np.nan
            self.result["curvature_mean"] = np.nan

    def _calculate_distance_correlation(self):
        """Calculates the correlation between graph-based distances and embedding-based distances.

        This metric assesses how well the embedding preserves the relative distances defined by the trajectory graph.
        It computes the shortest path lengths between all pairs of labels in the trajectory graph and the Euclidean
        distances between their corresponding centroids in the embedding space. The Spearman correlation between
        these two sets of distances is then calculated.

        Mathematical Definition:
            Distance Correlation = Spearmanr(D_graph, D_embedding)

        Where:
            - D_graph is the vector of shortest path lengths between all pairs of labels in the trajectory graph.
            - D_embedding is the vector of Euclidean distances between the corresponding centroids in the embedding.

        Advantages:
            - Measures monotonic relationships, providing robustness to non-linear scalings between graph and
                embedding distances.
            - Utilizes rank-order sensitivity, focusing on the order of distances rather than their absolute
                magnitudes, which is valuable for embeddings that might distort scales
                while preserving relative proximities.

        Limitations:
            - Assumes that biological trajectories represented in the graph are uniformly spaced,
                which may not hold true for all biological processes where some transitions might be naturally
                closer or more separated than others.
            - Assumes full connectivity within the graph; disconnected components can result in
                undefined shortest paths, complicating the correlation calculations.
            - Computationally intensive for large datasets, as it requires calculating pairwise
                distances for potentially large numbers of nodes.

        Sensitivities:
            - Sensitive to outliers in distance measurements, which can disproportionately influence the
                rank ordering and the resulting correlation, potentially skewing insights into the embedding's quality.
            - Dependent on pathfinding strategies in complex graphs, especially those with cycles or
                multiple routes, which can vary the computed graph distances significantly.

        Result:
            - A scalar value representing the Spearman correlation between graph and embedding distances.
        """
        try:
            longest_path = self._get_longest_path(self.prepared_after_subset_given)
            num_labels = len(longest_path)
            if num_labels < 2:
                self.logger.debug(f"Longest trajectory path has {num_labels} labels.")
                raise ValueError("Not enough labels in the trajectory to compute distance correlation.")
            centroids = self._get_centroids(longest_path)

            # Compute graph-based distances (shortest path lengths)
            # For all unique pairs in the longest path
            d_graph = []
            d_embedding = []
            for i in range(num_labels):
                for j in range(i + 1, num_labels):
                    label_i = longest_path[i]
                    label_j = longest_path[j]

                    # Shortest path length in the graph
                    try:
                        shortest_path_length = nx.shortest_path_length(
                            self.prepared_after_subset_given,
                            source=label_i,
                            target=label_j,
                            weight=None,  # Assuming unweighted graph; adjust if weighted
                        )
                    except nx.NetworkXNoPath:
                        self.logger.warning(
                            f"No path between '{label_i}' and '{label_j}' in the trajectory graph. Skipping this pair."
                        )
                        continue

                    d_graph.append(shortest_path_length)

                    # Euclidean distance in the embedding
                    centroid_i = centroids[i]
                    centroid_j = centroids[j]
                    euclidean_distance = np.linalg.norm(centroid_j - centroid_i)
                    d_embedding.append(euclidean_distance)

                    self.logger.debug(
                        f"Pair ('{label_i}', '{label_j}'): graph_distance = {shortest_path_length}, "
                        f"embedding_distance = {euclidean_distance}"
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
                self.logger.info(f"Distance Correlation (Spearman) calculated: {correlation} (p-value: {p_value})")

        except Exception as e:
            self.logger.error(f"Failed to calculate distance correlation: {e}")
            self.result["distance_correlation"] = np.nan

    def _helper_distances(self):
        # Extract all nodes involved in the trajectory graph
        nodes = list(self.prepared_after_subset_given.nodes())
        num_nodes = len(nodes)
        if num_nodes < 2:
            self.logger.debug(f"Trajectory graph has {num_nodes} nodes. Not enough to compute the metric.")
            raise ValueError("Not enough nodes in the trajectory graph to compute the metric.")

        self.logger.debug(f"Number of nodes in the trajectory graph: {num_nodes}")

        # Compute centroids for each node based on their labels
        centroids = self._get_centroids(nodes)
        centroids = np.array(centroids)

        # Compute embedding-based pairwise Euclidean distances using scipy's pdist
        embedding_distances = pdist(centroids, metric="euclidean")
        self.logger.debug("Computed embedding-based pairwise Euclidean distances.")

        # Initialize a list to store graph-based pairwise shortest path distances
        graph_distances = []

        # Compute graph-based pairwise shortest path distances
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                node_i = nodes[i]
                node_j = nodes[j]
                try:
                    # Compute shortest path length; adjust 'weight' if your graph uses weighted edges
                    distance = nx.shortest_path_length(
                        self.prepared_after_subset_given,
                        source=node_i,
                        target=node_j,
                        weight="length",  # Change to None if the graph is unweighted
                    )
                    graph_distances.append(distance)
                    self.logger.debug(f"Pair ('{node_i}', '{node_j}'): graph_distance = {distance}")
                except nx.NetworkXNoPath:
                    # Handle disconnected pairs by excluding them
                    graph_distances.append(np.nan)
                    self.logger.warning(
                        f"No path between '{node_i}' and '{node_j}' in the trajectory graph. Skipping this pair."
                    )

        graph_distances = np.array(graph_distances)
        embedding_distances = np.array(embedding_distances)
        return graph_distances, embedding_distances

    def _calculate_stress(self):
        """Calculates the stress function between graph distances and embedding distances across the entire trajectory graph.

        The stress function quantifies the discrepancy between the pairwise shortest path distances
        in the trajectory graph and the Euclidean distances between the corresponding centroids
        in the embedding space. By considering all pairwise distances, including those arising from
        branching pathways, the metric provides a comprehensive assessment of how well the embedding
        preserves the trajectory's structural integrity.

        Mathematical Definition:
            Stress = \\sqrt{ \frac{ \\sum_{i < j} (d_{ij}^{\text{embedding}} - d_{ij}^{\text{graph}})^2 }{ \\sum_{i < j} (d_{ij}^{\text{graph}})^2 } }

        Where:
            - \\( d_{ij}^{\text{graph}} \\) is the shortest path distance between nodes \\( i \\) and \\( j \\) in the trajectory graph.
            - \\( d_{ij}^{\text{embedding}} \\) is the Euclidean distance between the centroids of nodes \\( i \\) and \\( j \\) in the embedding space.

        Advantages:
            - Comprehensive Structural Assessment: Evaluates the preservation of both the primary trajectory and its branching pathways.
            - Global and Local Sensitivity: Sensitive to discrepancies at all scales, ensuring that both global trajectory shape and local branch fidelity are assessed.
            - Scale-Invariant Comparison: Normalization mitigates the impact of differing scales between graph and embedding distances.

        Limitations:
            - Computational Complexity: Pairwise distance calculations scale quadratically with the number of trajectory nodes, which can be computationally intensive for large graphs.
            - Assumption of Accurate Graph Representation: Relies on the trajectory graph accurately representing biological transitions, which may not capture all biological nuances.
            - Exclusion of Disconnected Pairs: Node pairs without a valid path in the graph are excluded, which can bias the stress value if many pairs are omitted.

        Sensitivities:
            - Outlier Influence: Extreme discrepancies between graph and embedding distances can disproportionately elevate the stress value.
            - Centroid Accuracy: High variability within cell-type labels can lead to inaccurate centroid representations, affecting distance calculations.
            - Pathfinding Strategy: The method assumes that the shortest path in the graph is biologically meaningful; alternative path definitions may be necessary for different biological contexts.

        Biological Considerations:
            - Non-Uniform Transition Rates: Biological processes often involve non-uniform state transitions, where some transitions occur more rapidly or are less distinct. The stress metric captures these variations by evaluating all pairwise distances.
            - Handling Cellular Heterogeneity: By computing centroids for each cell-type label, the method accounts for cellular heterogeneity, though high variability within labels may necessitate alternative representative measures.

        Result:
            - Scalar Stress Value: A float representing the stress, where lower values denote better preservation of the trajectory structure in the embedding. Values closer to 0 indicate minimal distortion, while higher values signify greater discrepancies.
        """
        try:
            graph_distances, embedding_distances = self._helper_distances()

            # Remove pairs with undefined graph distances (i.e., np.nan)
            valid_mask = ~np.isnan(graph_distances)
            num_valid_pairs = np.sum(valid_mask)
            self.logger.debug(
                f"Number of valid pairs for stress calculation: {num_valid_pairs} out of {len(graph_distances)}"
            )

            if num_valid_pairs == 0:
                stress = np.nan
                self.logger.warning("No valid graph distances available for stress calculation.")
            else:
                # Select only valid pairs
                graph_distances = graph_distances[valid_mask]
                embedding_distances = embedding_distances[valid_mask]

                # Normalize distances to mitigate scaling differences
                graph_max = np.max(graph_distances)
                embedding_max = np.max(embedding_distances)

                if graph_max == 0 or embedding_max == 0:
                    self.logger.warning("Maximum graph or embedding distance is zero. Cannot normalize distances.")
                    stress = np.nan
                else:
                    graph_distances_norm = graph_distances / graph_max
                    embedding_distances_norm = embedding_distances / embedding_max

                    # Compute stress numerator and denominator
                    stress_numerator = np.sum((embedding_distances_norm - graph_distances_norm) ** 2)
                    stress_denominator = np.sum(graph_distances_norm**2)

                    if stress_denominator == 0:
                        self.logger.warning("Denominator in stress calculation is zero. Cannot compute stress.")
                        stress = np.nan
                    else:
                        stress = np.sqrt(stress_numerator / stress_denominator)
                        self.logger.debug(f"Stress function: {stress}")

            self.result["stress"] = stress

        except Exception as e:
            self.logger.error(f"Failed to calculate stress: {e}")
            self.result["stress"] = np.nan

    def _calculate_geodesic_distance_correlation(self):
        """Calculates the Pearson between graph geodesic distances and embedding Euclidean distances.

        This metric assesses the linear relationship between the shortest path lengths in the
        trajectory graph (geodesic distances) and the Euclidean distances between the corresponding
        centroids in the embedding space. A high positive correlation indicates that the embedding
        preserves the intrinsic geometry of the trajectory graph, whereas a low or negative
        correlation suggests discrepancies between the graph structure and the embedding.

        Mathematical Definition:
            Let G = (V, E) be the trajectory graph, and let E = {e_1, e_2, ..., e_m} be the edges.
            Let d_G(i, j) denote the shortest path length between nodes i and j in G.
            Let d_E(i, j) denote the Euclidean distance between the centroids of nodes i and j
            in the embedding space.

            Pearson Correlation Coefficient:
                r = Cov(d_G, d_E) / (sigma_{d_G} * sigma_{d_E})

            Where:
                - Cov(d_G, d_E) is the covariance between the graph and embedding distances.
                - sigma_{d_G} and sigma_{d_E} are the standard deviations of the graph and embedding distances,
                  respectively.

        Advantages:
            - Linear Relationship Assessment: Specifically targets linear associations, providing
              clear interpretability of how well the embedding preserves the trajectory's geometry.
            - Simplicity and Efficiency: Computationally efficient, leveraging optimized statistical
              functions from SciPy.
            - Sensitivity to Scale and Translation: Since Pearson correlation is scale-invariant,
              it effectively measures the alignment of distance patterns regardless of the absolute
              scaling in the embedding space.

        Limitations:
            - Linear Assumption: Only captures linear relationships, potentially overlooking
              non-linear preservations or distortions in the embedding.
            - Sensitivity to Outliers: Extreme distance values can disproportionately influence
              the correlation coefficient, potentially skewing the metric.
            - Requires Complete Connectivity: Assumes that all node pairs are connected in the
              trajectory graph. Disconnected pairs are excluded, which might bias the correlation if
              a significant number of pairs are omitted.

        Preconditions:
            - The trajectory graph should be connected to ensure meaningful geodesic distances.
            - The embedding should provide a centroid for each node label present in the graph.

        Raises:
            - ValueError: If there are insufficient nodes to compute the correlation or if centroids
              cannot be computed for all nodes.
        """
        try:
            graph_distances, embedding_distances = self._helper_distances()

            # Remove pairs with undefined graph distances (i.e., np.nan)
            valid_mask = ~np.isnan(graph_distances)
            num_valid_pairs = np.sum(valid_mask)
            self.logger.debug(
                f"Number of valid pairs for geodesic distance correlation: {num_valid_pairs} out of {len(graph_distances)}"
            )

            if num_valid_pairs < 2:
                self.logger.warning("Insufficient valid pairs to compute Pearson correlation.")
                self.result["geodesic_distance_correlation"] = np.nan
                return

            # Select only valid pairs
            graph_distances_valid = graph_distances[valid_mask]
            embedding_distances_valid = embedding_distances[valid_mask]

            # Check for constant distance vectors which would result in undefined correlation
            if np.std(graph_distances_valid) == 0 or np.std(embedding_distances_valid) == 0:
                self.logger.warning("One of the distance vectors is constant. Pearson correlation is undefined.")
                self.result["geodesic_distance_correlation"] = np.nan
                return

            # Compute Pearson correlation
            correlation, p_value = pearsonr(graph_distances_valid, embedding_distances_valid)

            if np.isnan(correlation):
                self.logger.warning("Pearson correlation resulted in NaN. Possibly due to constant distance vectors.")
                self.result["geodesic_distance_correlation"] = np.nan
            else:
                self.result["geodesic_distance_correlation"] = correlation
                self.logger.info(f"Geodesic Distance Correlation (Pearson): {correlation:.4f} (p-value: {p_value:.4e})")

        except Exception as e:
            self.logger.error(f"Failed to calculate geodesic distance correlation: {e}")
            self.result["geodesic_distance_correlation"] = np.nan


## TODO: as embedding metric, Local Neighborhood Preservation (Graph-Adjacency vs. Embedding k-NN)

# Motivation
# In a well-preserved trajectory, cells (or labels) that are neighbors (or closely connected) in the trajectory graph should lie close to one another in the embedding.
# It captures how well local connectivity or adjacency structure is maintained, rather than focusing solely on large-scale or global paths.

# Method
# For each node (e.g., label or cell) in the trajectory graph, identify its nearest neighbors in the graph (e.g., immediate successors, BFS neighbors, etc.).
# In the embedding space, compute the k-nearest neighbors for the same node.
# Compare the overlap of these two sets (e.g., via Jaccard index, ) or any set-similarity measure.
# Aggregate the overlap score across all nodes to get a final “local neighborhood preservation” score.

# Advantages
# Sensitive to local structure.
# Easy to interpret and implement.

# Limitations
# Choice of  matters (e.g., how many neighbors to consider).
# Does not explicitly measure how “far” or “close” the neighbors are in the embedding—only membership overlap.

# Biological Relevance
# Cells that are functionally or developmentally close in a trajectory graph should remain close in the embedding if the embedding is biologically meaningful.


## TODO: as embedding metric, Sammon’s Mapping Error (Sammon Stress)

# Motivation
# A classical multidimensional scaling (MDS)-like measure that focuses on pairwise distances, but weights smaller distances more heavily.
# Complements the “stress” metric I already implemented but can give different weightings to large vs. small distances.

# Method Sketch
# Let  be the distance in the graph (or geodesic distance, or even direct edge distance if relevant).
# Let  be the Euclidean distance in the embedding.
# The Sammon stress is often given by:
# The division by  penalizes errors in short distances more than long distances.

# Advantages
# Highlights local distortions, which are crucial for trajectory “smoothness.”
# Good if you consider smaller distances in the graph (local transitions) to be critical to preserve.

# Limitations
# More sensitive to noise in small distances; can be disproportionately affected by measurement errors on small edges.
# Computationally  in the number of nodes (like typical distance-based metrics).

# Biological Relevance
# For developmental or differentiation trajectories, preserving local neighbor relationships (early vs. late states) can be more important than large jumps.


## TODO: as embedding metric, Branch Preservation Index

# Motivation
# Trajectories in biology often branch (e.g., cell fate decisions). A single path metric (like curvature) may miss whether branch points in the graph remain branch points in the embedding.

# Method
# Identify branch nodes (i.e., nodes with outdegree > 1) in the trajectory graph.
# In the embedding, examine the local distribution of points/centroids around those branch nodes:
# Compute angles between successive branches.
# Or measure cluster separability among different branches stemming from the same node.
# Define an index that quantifies how well separate branches remain separated or diverge in the embedding. For example:
# and then average over all branch nodes .

# Advantages
# Directly checks whether the embedding differentiates branching fates (important in many biological differentiation contexts).
# Captures local structural properties that might be lost in global metrics.

# Limitations
# Requires a clear definition of branches and branch nodes in the graph.
# Sensitive to how you define or label branches at each node.

# Biological Relevance
# Essential when multiple cell fates diverge from a common progenitor (common in lineage tracing or single-cell differentiation pathways).


## TODO: as embedding metric, Gromov–Hausdorff or Gromov-like Metric

# Motivation
# Captures how similar two metric spaces are “up to isometry.” In principle, it tries to measure how close the trajectory graph’s metric is to the embedding’s metric in an isometric sense.
# A more mathematically rigorous measure of “distance between metric spaces” than simple stress or correlation.

# Method Sketch
# The Gromov–Hausdorff distance (GH) is defined as the minimum Hausdorff distance between the two metric spaces when embedded into a common metric space.
# Direct GH is complex to compute for large graphs. However, approximate or restricted versions can give a measure of how well the metric geometry of the graph is preserved.

# Advantages
# The gold-standard for comparing metric spaces “up to isometry.”
# Very general and captures subtle distortions.

# Limitations
# Computationally expensive in the general form.
# Might be overkill unless you have a smaller trajectory graph or can handle approximations.

# Biological Relevance
# If you want a very robust measure of how well the entire geometry is matched—useful in advanced topological or manifold-based trajectory analyses.


## TODO: as embedding metric, Neighborhood Preservation by Rank Correlation (Shepard Diagram Correlation)

# Motivation
# A “Shepard diagram” plots graph distances on one axis vs. embedding distances on the other. The Spearman or Pearson correlation of those points is a measure of rank or linear association.
# This is conceptually similar to your “distance_correlation” but is often reported specifically under the name “Shepard diagram correlation” in embedding literature.

# Method
# For each pair of nodes :
#  = geodesic distance in the graph.
#  = Euclidean distance in the embedding.
# Plot or compute correlation().
# A variation is to weight these distances by  to emphasize local structure.

# Advantages
# Well-known approach for evaluating dimensionality reduction.
# Easy to implement (just gather pairwise distances and compute correlation).

# Limitations
# Suffers from the same potential issues of heavy weighting or ignoring large distances, depending on the variant.
# Potentially  in the number of nodes.

# Biological Relevance
# Shows how well local and global distances are preserved in a single correlation measure. Useful if your graph-based distances approximate real biological dissimilarities.


# TODO: as embedding metric, Pairwise Transitions or Edge Preservation Score

# Motivation
# Sometimes you only need to test whether edges in the graph correspond to edges or close points in the embedding. This focuses specifically on direct connections rather than all-pairs.

# Method
# For every directed edge  in the trajectory graph, measure:
# The distance in the embedding between centroids (or cells) of  and .
# Possibly also measure the vector direction if the embedding is directional (e.g., velocity or time-lapse data).
# Summarize how many graph edges are “short” or “preserved” in the embedding compared to the average inter-node distances.

# Advantages
# Fast to compute if the graph is not too dense (only sums over edges, not all pairs).
# Useful if you specifically trust edges in the graph as “true transitions.”

# Limitations
# Ignores indirect relationships (i.e., length-2 or longer paths).
# If the graph is dense or has many edges, this can still be large.

# Biological Relevance
# If edges represent immediate biological transitions (e.g., from progenitor to intermediate state), this checks those transitions’ presence in the embedding.


# TODO: as embedding metric, Label Continuity / Label Transition Probability

# Motivation
# If your trajectory is labeled by known cell types or states, you might want to check whether states that should follow each other along the trajectory do so in the embedding.

# Method
# Given the trajectory edges , find the fraction of points (cells) in the embedding that truly “border” each other in the same direction—i.e., if you pick a cell labeled , do its nearest neighbors in the embedding mostly come from ?
# Define a continuity or transition probability measure, e.g.:
# Aggregate or compare to the graph connectivity to see if the embedding preserves that adjacency.

# Advantages
# Ties directly to biological labeling or known trajectory steps.
# Can detect if the embedding merges states that should remain distinct or incorrectly separates states that are adjacent.

# Limitations
# Requires prior labels or states in addition to the graph structure.
# Has some hyperparameters like the size of the local neighborhood in the embedding.

# Biological Relevance
# If the graph was built from known transitions (e.g., experimental lineage tracing), checking that those transitions appear “locally plausible” in the embedding is crucial.


# TODO: as embedding metric, Manifold Density Overlap or Coverage

# Motivation
# If the trajectory graph is supposed to occupy a sub-manifold or path in the embedding, we can see how well the embedding’s actual data points align or cover that manifold.

# Method Sketch
# Approximate the manifold or path defined by the trajectory in the embedding space (e.g., use a spline or principal curve that goes through the ordered centroids).
# Estimate the density of actual data points (cells) around that curve. For instance, for each point on the curve, measure how many real data points fall within a small radius .
# Aggregate to get a measure of coverage: if large portions of the curve have no real data points near them, the embedding might not faithfully reflect the trajectory or might have “bent” the path away from the data.

# Advantages
# Captures how well the trajectory path is actually “populated” by real cells, rather than being an abstract line that the embedding might not reflect well.
# Reveals if the embedding artificially stretches or shrinks parts of the path to isolate or cluster cells incorrectly.

# Limitations
# Requires a good curve-fitting approach in the embedding and a suitable radius or bandwidth for density estimation.
# Sensitive to outliers and data sampling variability.

# Biological Relevance
# If the trajectory is real, you expect continuous coverage of intermediate states in scRNA-seq or single-cell data along the entire path.
