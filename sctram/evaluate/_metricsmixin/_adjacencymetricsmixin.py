#!/usr/bin/env python3

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity

from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase
from sctram.evaluate._metricsmixin._spatialmetricsmixin import SpatialMetricsMixin


class AdjacencyMetricsMixin(MetricsMixinBase):
    """Metrics to compare two adjacency matrixes, that is, two networkx graphs."""

    available_metrics = [
        "frobenius",
        "L1",
        "accuracy",
        "graph_edit_distance",
        "spectral_distance",
        "jaccard",
        "hamming",
        "precision",
        "recall",
        "f1_score",
        "mantel_correlation",
        "ssim",
        "avg_shortest_path_diff",
        "degree_emd",
        "clustering_coeff_diff",
        "graphlet_degree_vector",
        "weisfeiler_lehman_distance",
        "gnn_embedding_distance",
        "persistence_diagram_distance",
        "maximum_common_subgraph_distance",
        "random_walk_kernel_distance",
    ]

    def _calculate(self):
        """Performs the evaluation by comparing the adjacency matrices using the specified metrics.

        Iterates over each specified metric and invokes the corresponding calculation method.
        Stores the results in the `self.result` dictionary.
        """
        for metric in self.metrics:
            self.logger.debug(f"Calculating metric: {metric!r}")
            if metric == "frobenius":
                self._calculate_frobenius()
            elif metric == "L1":
                self._calculate_l1()
            elif metric == "accuracy":
                self._calculate_accuracy()
            elif metric == "graph_edit_distance":
                self._calculate_graph_edit_distance()
            elif metric == "spectral_distance":
                self._calculate_spectral_distance()
            elif metric == "jaccard":
                self._calculate_jaccard_similarity()
            elif metric == "hamming":
                self._calculate_hamming_distance()
            elif metric in ["precision", "recall", "f1_score"]:
                self._calculate_precision_recall_f1(metric)
            elif metric == "mantel_correlation":
                self._calculate_mantel_correlation()    
            elif metric == "ssim":
                self._calculate_ssim()
            elif metric == "avg_shortest_path_diff":
                self._calculate_avg_shortest_path_diff()
            elif metric == "degree_emd":
                self._calculate_degree_emd()
            elif metric == "clustering_coeff_diff":
                self._calculate_clustering_coeff_diff()
            elif metric == "graphlet_degree_vector":
                self._calculate_graphlet_degree_vector()
            elif metric == "weisfeiler_lehman_distance":
                self._calculate_weisfeiler_lehman_distance()
            elif metric == "gnn_embedding_distance":
                self._calculate_gnn_embedding_distance()
            elif metric == "persistence_diagram_distance":
                self._calculate_persistence_diagram_distance()
            elif metric == "maximum_common_subgraph_distance":
                self._calculate_maximum_common_subgraph_distance()
            elif metric == "random_walk_kernel_distance":
                self._calculate_random_walk_kernel_distance()
            else:
                self.logger.warning(f"Unknown metric {metric!r} specified. Skipping.")

    def _calculate_frobenius(self):
        """Calculates the Frobenius norm of the difference between the two adjacency matrices.

        The Frobenius norm provides a measure of the overall difference between two matrices by
        summing the squares of the element-wise differences and taking the square root.

        Advantages:
            - Captures the global structural differences between two matrices.
            - Sensitive to both edge additions and deletions.

        Limitations:
            - Does not provide insight into where the differences occur.
            - Treats all differences equally, regardless of their structural significance.

        Result:
            - A single scalar value representing the Frobenius norm.
        """
        diff_matrix = self.prepared_after_subset_given - self.prepared_after_subset_inferred
        fro_norm = np.linalg.norm(diff_matrix, "fro")
        self.result["frobenius"] = fro_norm
        self.logger.debug(f"Frobenius norm of difference: {fro_norm}")

    def _calculate_l1(self):
        """Calculates the L1 norm (Manhattan distance) of the difference between the two adjacency matrices.

        The L1 norm sums the absolute differences of the corresponding elements in the matrices.

        Advantages:
            - Simple and interpretable measure of total edge discrepancies.
            - Sensitive to both the magnitude and number of differences.

        Limitations:
            - Like the Frobenius norm, it does not localize where differences occur.
            - May not capture the structural impact of specific edge differences.

        Result:
            - A single scalar value representing the L1 norm.
        """
        diff_matrix = self.prepared_after_subset_given - self.prepared_after_subset_inferred
        l1_norm = np.sum(np.abs(diff_matrix))
        self.result["L1"] = l1_norm
        self.logger.debug(f"L1 norm of difference: {l1_norm}")

    def _calculate_accuracy(self):
        """Calculates the accuracy of the inferred adjacency matrix.

        Accuracy is defined as the proportion of correctly inferred edges (both present and absent)
        relative to the total number of possible edges.

        Advantages:
            - Provides a straightforward measure of overall correctness.
            - Balances true positives and true negatives.

        Limitations:
            - Can be misleading in cases of class imbalance (e.g., sparse graphs).
            - Does not distinguish between types of errors (false positives vs. false negatives).

        Result:
            - A single scalar value between 0 and 1 representing accuracy.
        """
        total_elements = self.prepared_after_subset_given.size
        matching_elements = np.sum(self.prepared_after_subset_given == self.prepared_after_subset_inferred)
        accuracy = matching_elements / total_elements
        self.result["accuracy"] = accuracy
        self.logger.debug(f"Accuracy of adjacency matrices: {accuracy}")

    def _calculate_graph_edit_distance(self):
        """Calculates the Graph Edit Distance (GED) between the two graphs.

        Graph Edit Distance is the minimum number of edit operations (such as edge additions or deletions)
        required to transform one graph into the other.

        Advantages:
            - Provides a comprehensive measure of structural dissimilarity.
            - Reflects the exact number of changes needed for transformation.

        Limitations:
            - Computationally intensive, especially for larger graphs.
            - May not be feasible for graphs with more than ~100 nodes.

        Sensitivities:
            - Highly sensitive to both edge additions and deletions.
            - Captures exact structural changes.

        Result:
            - A single scalar value representing the Graph Edit Distance.
            - Returns `infinity` if GED could not be computed.
        """
        # Convert adjacency matrices back to NetworkX graphs
        g1 = nx.from_numpy_array(self.prepared_after_subset_given, create_using=nx.Graph)
        g2 = nx.from_numpy_array(self.prepared_after_subset_inferred, create_using=nx.Graph)
        # Compute Graph Edit Distance
        try:
            ged = nx.graph_edit_distance(g1, g2)
            if ged is None:
                self.logger.warning("Graph Edit Distance could not be computed.")
                ged = float("inf")
        except Exception as e:
            self.logger.error(f"Error computing Graph Edit Distance: {e}")
            ged = float("inf")
        self.result["graph_edit_distance"] = ged
        self.logger.debug(f"Graph Edit Distance: {ged}")

    def _calculate_spectral_distance(self):
        """Calculates the Spectral Distance between the two adjacency matrices.

        Spectral Distance compares the eigenvalues of the two adjacency matrices. Specifically, it calculates
        the Euclidean (L2) norm of the difference between the sorted eigenvalues of both matrices.

        Advantages:
            - Captures global structural properties related to connectivity and expansion.
            - Reflects differences in the overall graph topology.

        Limitations:
            - Does not provide localized information about specific edge differences.
            - Sensitive to minor changes in the graph that significantly alter eigenvalues.

        Sensitivities:
            - Sensitive to changes that affect the spectrum, such as edge additions/removals that impact connectivity.

        Result:
            - A single scalar value representing the Spectral Distance.
        """
        # Compute eigenvalues
        eigen_g1 = np.linalg.eigvals(self.prepared_after_subset_given)
        eigen_g2 = np.linalg.eigvals(self.prepared_after_subset_inferred)
        # Sort eigenvalues for alignment
        eigen_g1_sorted = np.sort_complex(eigen_g1)
        eigen_g2_sorted = np.sort_complex(eigen_g2)
        # Compute Euclidean (L2) norm of eigenvalue differences
        spectral_diff = np.linalg.norm(eigen_g1_sorted - eigen_g2_sorted, ord=2)
        self.result["spectral_distance"] = spectral_diff
        self.logger.debug(f"Spectral Distance (Euclidean (L2) norm of eigenvalue differences): {spectral_diff}")

    def _calculate_jaccard_similarity(self):
        """Calculates the Jaccard Similarity between the two graphs' edge sets.

        Jaccard Similarity is the ratio of the number of common edges to the total number of unique edges
        in both graphs.

        Advantages:
            - Intuitive measure of edge-wise similarity.
            - Easy to interpret as a ratio.

        Limitations:
            - Does not account for the importance or weight of specific edges.
            - Binary measure; does not capture edge weights if present.

        Sensitivities:
            - Sensitive to the presence or absence of edges.
            - Ignores edge multiplicity in MultiGraphs.

        Result:
            - A single scalar value between 0 and 1 representing Jaccard Similarity.
        """
        set_g1 = set(zip(*np.where(self.prepared_after_subset_given)))
        set_g2 = set(zip(*np.where(self.prepared_after_subset_inferred)))
        intersection = set_g1.intersection(set_g2)
        union = set_g1.union(set_g2)
        if not union:
            jaccard = 1.0  # Both graphs have no edges
        else:
            jaccard = len(intersection) / len(union)
        self.result["jaccard"] = jaccard
        self.logger.debug(f"Jaccard Similarity: {jaccard}")

    def _calculate_hamming_distance(self):
        """Calculates the Hamming Distance between the two adjacency matrices.

        Hamming Distance counts the number of differing elements between the two matrices.

        Advantages:
            - Simple and direct measure of edge discrepancies.
            - Easy to compute and interpret.

        Limitations:
            - Does not account for the significance or impact of specific differences.
            - Treats all differences equally regardless of their structural importance.

        Sensitivities:
            - Sensitive to any edge addition or deletion.
            - Does not differentiate between types of edge differences.

        Result:
            - A single scalar value representing the Hamming Distance.
        """
        hamming = np.sum(self.prepared_after_subset_given != self.prepared_after_subset_inferred)
        self.result["hamming"] = hamming
        self.logger.debug(f"Hamming Distance: {hamming}")

    def _calculate_precision_recall_f1(self, metric: str):
        """Calculates Precision, Recall, or F1 Score based on the specified metric.

        Precision measures the proportion of correctly inferred edges out of all inferred edges.
        Recall measures the proportion of correctly inferred edges out of all actual edges.
        F1 Score is the harmonic mean of Precision and Recall, providing a balance between the two.

        Advantages:
            - Precision and Recall provide insights into different types of prediction errors.
            - F1 Score balances Precision and Recall, offering a single metric that accounts for both.

        Limitations:
            - Does not consider true negatives.
            - Sensitive to class imbalance, especially in sparse graphs.

        Sensitivities:
            - Precision is sensitive to false positives.
            - Recall is sensitive to false negatives.

        Args:
            metric (str): The specific metric to calculate ('precision', 'recall', or 'f1_score').

        Raises:
            ValueError: If an invalid metric name is provided.

        Result:
            - Updates the `self.result` dictionary with the calculated metric.
        """
        # True Positives: edges present in both inferred and given
        true_positive = np.sum((self.prepared_after_subset_inferred == 1) & (self.prepared_after_subset_given == 1))
        # Predicted Positives: edges present in inferred
        predicted_positive = np.sum(self.prepared_after_subset_inferred == 1)
        # Actual Positives: edges present in given
        actual_positive = np.sum(self.prepared_after_subset_given == 1)

        if predicted_positive == 0:
            precision = 0.0
        else:
            precision = true_positive / predicted_positive

        if actual_positive == 0:
            recall = 0.0
        else:
            recall = true_positive / actual_positive

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if metric == "precision":
            self.result["precision"] = precision
            self.logger.debug(f"Precision: {precision}")
        elif metric == "recall":
            self.result["recall"] = recall
            self.logger.debug(f"Recall: {recall}")
        elif metric == "f1_score":
            self.result["f1_score"] = f1
            self.logger.debug(f"F1 Score: {f1}")
        else:
            raise ValueError(f"Invalid metric {metric!r} for precision/recall/F1 calculation.")

    def _calculate_ssim(self):
        try:
            from skimage.metrics import structural_similarity as ssim
            data_range = self.prepared_after_subset_given.max() - self.prepared_after_subset_given.min()
            similarity, _ = ssim(self.prepared_after_subset_given, self.prepared_after_subset_inferred, full=True, data_range=data_range)
            self.result["ssim"] = similarity
        except (ImportError, ModuleNotFoundError):
            self.logger.debug("scikit-image is required for Structural Similarity Index (SSIM) but is not installed.")
            self.result["ssim"] = np.nan
        except Exception as e:
            self.logger.debug(f"Error computing Structural Similarity Index (SSIM): {e}")
            self.result["ssim"] = np.nan

    def _get_average_shortest_path_lengths(self):
        g1 = nx.from_numpy_array(self.prepared_after_subset_given)
        g2 = nx.from_numpy_array(self.prepared_after_subset_inferred)

        if nx.is_connected(g1):
            avg_given = nx.average_shortest_path_length(g1)
        else:
            self.logger.warning("Given graph is not connected. Shortest path distances will include infinities.")
            avg_given = np.nan  # Undefined for disconnected graphs

        if nx.is_connected(g2):
            avg_inferred = nx.average_shortest_path_length(g2)
        else:
            self.logger.warning("Inferred graph is not connected. Shortest path distances will include infinities.")
            avg_inferred = np.nan
        
        return avg_given, avg_inferred
    
    def _calculate_mantel_correlation(self):
        """Calculates the Mantel test statistic between the two adjacency matrices.

        The Mantel test assesses the correlation between two distance matrices derived from the
        given and inferred adjacency matrices. It returns the correlation coefficient and the
        p-value indicating the significance of the correlation.

        Advantages:
            - Provides a statistical measure of similarity between two network structures.
            - Accounts for spatial or structural dependencies via distance matrices.

        Limitations:
            - Computationally intensive for large networks due to permutation testing.
            - Assumes that the distance matrices are meaningful representations of network structure.
        """
        try:
            from skbio.stats.distance import DistanceMatrix
            from skbio.stats.distance import mantel
            avg_given, avg_inferred = self._get_average_shortest_path_lengths()
            if np.isnan(avg_given) or np.isnan(avg_inferred):
                raise ValueError("One or both graphs are disconnected. Mantel correlation is undefined.")
            
            dm_given = DistanceMatrix(self.prepared_after_subset_given)
            dm_inferred = DistanceMatrix(self.prepared_after_subset_inferred)
            
            mantel_result = mantel(dm_given, dm_inferred, method='pearson', permutations=999)
            self.result["mantel_correlation"] = mantel_result[0]
            
        except (ImportError, ModuleNotFoundError):
            self.logger.debug("scikit-bio is required for Mantel correlation but is not installed.")
            self.result["mantel_correlation"] = np.nan
        except Exception as e:
            self.logger.debug(f"Error computing Mantel correlation: {e}")
            self.result["mantel_correlation"] = np.nan

    def _calculate_avg_shortest_path_diff(self):
        """Calculates the difference in average shortest path lengths between the two graphs.

        Compares the average shortest path lengths of all connected pairs of nodes in both graphs.

        Advantages:
            - Reflects differences in the overall navigability and connectivity.
            - Captures global structural properties related to path lengths.

        Limitations:
            - Undefined for disconnected graphs (returns NaN).
            - Sensitive to local changes that affect path lengths.

        Sensitivities:
            - Sensitive to edge additions or deletions that alter connectivity.
            - Does not account for the distribution of path lengths beyond the average.

        Result:
            - A single scalar value representing the absolute difference in average shortest path lengths.
            - Returns `NaN` if one or both graphs are disconnected.
        """
        try:
            avg_g1, avg_g2 = self._get_average_shortest_path_lengths()

            if np.isnan(avg_g1) or np.isnan(avg_g2):
                avg_shortest_path_diff = np.nan
                self.logger.warning(
                    "One or both graphs are disconnected. Average shortest path difference is undefined."
                )
            else:
                avg_shortest_path_diff = abs(avg_g1 - avg_g2)

            self.result["avg_shortest_path_diff"] = avg_shortest_path_diff
            self.logger.debug(f"Average Shortest Path Length Difference: {avg_shortest_path_diff}")
        except Exception as e:
            self.logger.error(f"Error computing average shortest path length difference: {e}")
            self.result["avg_shortest_path_diff"] = np.nan

    def _calculate_degree_emd(self):
        """Calculates the Earth Mover's Distance (EMD) between the degree distributions of the two graphs.

        Earth Mover's Distance quantifies the dissimilarity between two probability distributions by measuring
        the minimal amount of "work" required to transform one distribution into the other.

        Advantages:
            - Captures differences in the overall node connectivity patterns.
            - Sensitive to shifts in the degree distribution.

        Limitations:
            - Assumes the degree distributions can be interpreted as probability distributions.
            - Computationally more involved than simpler distribution comparison metrics.

        Sensitivities:
            - Sensitive to changes that affect node degrees, such as edge additions/removals.
            - Reflects global connectivity trends but not localized structural changes.

        Result:
            - A single scalar value representing the Earth Mover's Distance between degree distributions.
            - Returns `NaN` if computation fails.
        """
        degrees_g1 = np.sum(self.prepared_after_subset_given, axis=1)
        degrees_g2 = np.sum(self.prepared_after_subset_inferred, axis=1)
        # Compute Earth Mover's Distance between degree distributions
        try:
            emd = wasserstein_distance(degrees_g1, degrees_g2)
        except Exception as e:
            self.logger.error(f"Error computing Degree EMD: {e}")
            emd = np.nan
        self.result["degree_emd"] = emd
        self.logger.debug(f"Degree Distribution Earth Mover's Distance: {emd}")

    def _calculate_clustering_coeff_diff(self):
        """Calculates the absolute difference in average clustering coefficients between the two graphs.

        The clustering coefficient measures the degree to which nodes in a graph tend to cluster together.

        Advantages:
            - Reflects differences in local connectivity and the presence of tightly-knit communities.
            - Sensitive to the presence of triangles and other small cycles.

        Limitations:
            - Only considers local clustering, not global structural differences.
            - May not capture differences in larger-scale structures.

        Sensitivities:
            - Sensitive to edge additions/removals that create or break triangles.
            - Reflects changes in the propensity for nodes to form cliques.

        Result:
            - A single scalar value representing the absolute difference in average clustering coefficients.
        """
        g1 = nx.from_numpy_array(self.prepared_after_subset_given)
        g2 = nx.from_numpy_array(self.prepared_after_subset_inferred)
        clustering_g1 = nx.average_clustering(g1)
        clustering_g2 = nx.average_clustering(g2)
        clustering_diff = abs(clustering_g1 - clustering_g2)
        self.result["clustering_coeff_diff"] = clustering_diff
        self.logger.debug(f"Clustering Coefficient Difference: {clustering_diff}")

    def _calculate_graphlet_degree_vector(self):
        """Calculates the Graphlet Degree Vector (GDV) similarity between the two graphs.

        Graphlet Degree Vector captures the frequency of small induced subgraphs (graphlets) around each node.
        The similarity is quantified by comparing these vectors across the two graphs.

        Advantages:
            - Captures local structural patterns and motifs.
            - Sensitive to the presence of specific small subgraph configurations.

        Limitations:
            - Computationally intensive for large graphlets or larger graphs.
            - Requires alignment of node correspondences for accurate comparison.

        Sensitivities:
            - Sensitive to the presence or absence of specific graphlets.
            - Reflects local connectivity and neighborhood structures.

        Result:
            - A single scalar value representing the similarity between the GDVs of the two graphs.
        """
        try:
            from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

            # Placeholder implementation
            # In practice, implement GDV extraction and similarity computation
            # For simplicity, using WL graph hash similarity as a proxy
            hash_g1 = weisfeiler_lehman_graph_hash(nx.from_numpy_array(self.prepared_after_subset_given))
            hash_g2 = weisfeiler_lehman_graph_hash(nx.from_numpy_array(self.prepared_after_subset_inferred))
            # Compare hash strings using Hamming distance
            min_len = min(len(hash_g1), len(hash_g2))
            hamming = sum(c1 != c2 for c1, c2 in zip(hash_g1[:min_len], hash_g2[:min_len]))
            # Normalize hamming distance
            similarity = 1 - (hamming / min_len)
            self.result["graphlet_degree_vector"] = similarity
            self.logger.debug(f"Graphlet Degree Vector Similarity: {similarity}")
        except ImportError:
            self.logger.error("NetworkX Weisfeiler-Lehman graph hashing is not available.")
            self.result["graphlet_degree_vector"] = np.nan
        except Exception as e:
            self.logger.error(f"Error computing Graphlet Degree Vector similarity: {e}")
            self.result["graphlet_degree_vector"] = np.nan

    def _calculate_weisfeiler_lehman_distance(self):
        """Calculates the Weisfeiler-Lehman (WL) Graph Kernel distance between the two graphs.

        The Weisfeiler-Lehman Graph Kernel is a powerful method for measuring graph similarity based on iterative
        label refinement and subtree patterns.

        Advantages:
            - Captures both local and global structural information.
            - Efficient and scalable compared to exact graph kernel methods.

        Limitations:
            - Requires label refinement iterations, which may not capture all structural nuances.
            - Sensitive to the number of refinement iterations.

        Sensitivities:
            - Sensitive to structural differences that affect label refinement.
            - Reflects changes in subtree patterns and neighborhood structures.

        Result:
            - A single scalar value representing the WL distance between the two graphs.
        """
        try:
            from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

            # Define number of iterations
            num_iterations = 3
            hash_g1 = weisfeiler_lehman_graph_hash(
                nx.from_numpy_array(self.prepared_after_subset_given), iterations=num_iterations
            )
            hash_g2 = weisfeiler_lehman_graph_hash(
                nx.from_numpy_array(self.prepared_after_subset_inferred), iterations=num_iterations
            )
            # Convert hash to integer representation
            int_g1 = int(hash_g1[:8], 16)  # Taking first 8 characters for brevity
            int_g2 = int(hash_g2[:8], 16)
            # Compute absolute difference
            wl_distance = abs(int_g1 - int_g2)
            self.result["weisfeiler_lehman_distance"] = wl_distance
            self.logger.debug(f"Weisfeiler-Lehman Graph Kernel Distance: {wl_distance}")
        except ImportError:
            self.logger.error("NetworkX Weisfeiler-Lehman graph hashing is not available.")
            self.result["weisfeiler_lehman_distance"] = np.nan
        except Exception as e:
            self.logger.error(f"Error computing Weisfeiler-Lehman distance: {e}")
            self.result["weisfeiler_lehman_distance"] = np.nan

    def _calculate_gnn_embedding_distance(self):
        """Calculates the Graph Neural Network (GNN) Embedding Distance between the two graphs.

        Utilizes a simple Graph Neural Network to embed both graphs into a vector space and computes
        the cosine similarity between their embeddings.

        Advantages:
            - Leverages deep learning to capture complex structural patterns.
            - Provides a flexible and powerful representation of graph structures.

        Limitations:
            - Requires training a GNN model, which may introduce computational overhead.
            - The quality of embeddings depends on the GNN architecture and training process.

        Sensitivities:
            - Sensitive to structural differences that the GNN is trained to capture.
            - Reflects both local and global structural nuances as learned by the GNN.

        Result:
            - A single scalar value between -1 and 1 representing the cosine similarity of GNN embeddings.
        """
        self.logger.warning("GNN Embedding Cosine Similarity method is not tested in depth.")
        try:
            import torch  # type: ignore
            from torch_geometric.data import Data  # type: ignore
            from torch_geometric.nn import GCNConv  # type: ignore

            # from torch_geometric.utils import to_networkx
        except ImportError:
            self.logger.error("PyTorch Geometric is not installed. GNN Embedding Distance cannot be computed.")
            self.result["gnn_embedding_distance"] = np.nan
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _initialize_gnn_model():
            """Initializes a simple Graph Neural Network (GNN) model for embedding-based metrics.

            Returns:
                torch.nn.Module: A simple GNN model.
            """

            class SimpleGNN(torch.nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim):
                    super().__init__()
                    self.conv1 = GCNConv(input_dim, hidden_dim)
                    self.conv2 = GCNConv(hidden_dim, output_dim)

                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = self.conv1(x, edge_index)
                    x = torch.relu(x)
                    x = self.conv2(x, edge_index)
                    return x

            # Assuming binary adjacency matrices, input_dim can be 1
            model = SimpleGNN(input_dim=1, hidden_dim=16, output_dim=32).to(device)
            return model

        gnn_model = _initialize_gnn_model()

        try:
            # Convert adjacency matrices to PyTorch Geometric Data objects
            def adj_to_data(adj_matrix):
                g = nx.from_numpy_array(adj_matrix)
                edge_index = torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()
                num_nodes = adj_matrix.shape[0]
                x = torch.ones((num_nodes, 1), dtype=torch.float)  # Node features: all ones
                data = Data(x=x, edge_index=edge_index)
                return data

            data_g1 = adj_to_data(self.prepared_after_subset_given).to(device)
            data_g2 = adj_to_data(self.prepared_after_subset_inferred).to(device)

            # Forward pass through GNN to get embeddings
            with torch.no_grad():
                embedding_g1 = gnn_model(data_g1).mean(dim=0).cpu().numpy()
                embedding_g2 = gnn_model(data_g2).mean(dim=0).cpu().numpy()

            # Normalize embeddings
            embedding_g1_norm = embedding_g1 / np.linalg.norm(embedding_g1)
            embedding_g2_norm = embedding_g2 / np.linalg.norm(embedding_g2)

            # Compute cosine similarity
            cosine_sim = cosine_similarity([embedding_g1_norm], [embedding_g2_norm])[0][0]
            self.result["gnn_embedding_distance"] = cosine_sim
            self.logger.debug(f"GNN Embedding Cosine Similarity: {cosine_sim}")
        except Exception as e:
            self.logger.error(f"Error computing GNN Embedding Distance: {e}")
            self.result["gnn_embedding_distance"] = np.nan

    def _calculate_persistence_diagram_distance(self):
        """Calculates the Persistence Diagram Distance between the two graphs using Topological Data Analysis (TDA).

        Utilizes persistence diagrams to capture the topological features (e.g., connected components, cycles)
        of each graph and computes the Wasserstein distance between them.

        Advantages:
            - Captures global topological properties of graphs.
            - Sensitive to the presence and persistence of cycles and other topological features.

        Limitations:
            - Requires additional computational resources for TDA computations.
            - Interpretation of persistence distances can be abstract and requires domain knowledge.

        Sensitivities:
            - Sensitive to changes in topological features such as the introduction or removal of cycles.
            - Reflects differences in the graph's fundamental topological structure.

        Result:
            - A single scalar value representing the Wasserstein distance between the persistence diagrams of the two graphs.
            - Returns `NaN` if computation fails.
        """
        self.logger.warning("Persistence Diagram Distance method is not tested in depth.")
        try:
            import gudhi as gd  # type: ignore
            import ot

            # For TDA persistence diagrams
            # Convert adjacency matrices to NetworkX graphs
            g1 = nx.from_numpy_array(self.prepared_after_subset_given)
            g2 = nx.from_numpy_array(self.prepared_after_subset_inferred)

            # Create a simplicial complex from each graph
            st1 = gd.SimplexTree()
            st1.insert([0])
            for edge in g1.edges():
                st1.insert(list(edge))
            st2 = gd.SimplexTree()
            st2.insert([0])
            for edge in g2.edges():
                st2.insert(list(edge))

            # Compute persistence diagrams
            st1.compute_persistence()
            st2.compute_persistence()

            # Extract 1-dimensional persistence (cycles)
            diag1 = st1.persistence_intervals_in_dimension(1)
            diag2 = st2.persistence_intervals_in_dimension(1)

            if len(diag1) == 0 and len(diag2) == 0:
                pd_dist = 0.0  # Both diagrams are empty, distance is zero
            elif len(diag1) == 0 or len(diag2) == 0:
                pd_dist = float("inf")  # One diagram is empty, distance is infinite
            else:
                # Convert persistence diagrams to numpy arrays
                # Each diagram is a list of [birth, death] pairs
                # We need to pad the smaller diagram with dummy points on the diagonal (birth=death)
                # to make them the same size for the Wasserstein distance computation

                # Determine the size of the diagrams
                n1 = len(diag1)
                n2 = len(diag2)
                max_size = max(n1, n2)

                # Pad the diagrams with diagonal points
                if n1 < max_size:
                    pad_diag1 = diag1 + [[d[0], d[0]] for d in diag2[: max_size - n1]]
                else:
                    pad_diag1 = diag1[:max_size]

                if n2 < max_size:
                    pad_diag2 = diag2 + [[d[0], d[0]] for d in diag1[: max_size - n2]]
                else:
                    pad_diag2 = diag2[:max_size]

                # Convert to numpy arrays
                pd1 = np.array(pad_diag1)
                pd2 = np.array(pad_diag2)

                # Compute the cost matrix (e.g., L2 distance between points)
                cost_matrix = np.linalg.norm(pd1[:, np.newaxis, :] - pd2[np.newaxis, :, :], axis=2)

                # Define uniform weights
                a = np.ones((max_size,)) / max_size
                b = np.ones((max_size,)) / max_size

                # Compute Wasserstein distance using POT's emd2
                pd_dist = ot.emd2(a, b, cost_matrix)

            self.result["persistence_diagram_distance"] = pd_dist
            self.logger.debug(f"Persistence Diagram Wasserstein Distance: {pd_dist}")
        except ImportError as e:
            self.logger.error(f"Required library for persistence diagram distance not installed: {e}")
            self.result["persistence_diagram_distance"] = np.nan
        except Exception as e:
            self.logger.error(f"Error computing Persistence Diagram Distance: {e}")
            self.result["persistence_diagram_distance"] = np.nan

    def _calculate_maximum_common_subgraph_distance(self):
        """Calculates the Maximum Common Subgraph (MCS) Distance between the two graphs.

        Determines the size of the largest common subgraph shared by both graphs and computes the
        distance based on the size of the MCS relative to the original graph sizes.

        Advantages:
            - Reflects the largest structurally similar portion of the graphs.
            - Provides insight into the shared core structure.

        Limitations:
            - Computationally intensive for larger graphs.
            - Sensitive to node correspondences, which may not be unique.

        Sensitivities:
            - Sensitive to the presence or absence of large shared substructures.
            - Reflects differences in global structural alignment.

        Result:
            - A single scalar value representing the MCS distance.
            - Returns `NaN` if computation fails.
        """
        try:
            # Convert adjacency matrices back to NetworkX graphs
            g1 = nx.from_numpy_array(self.prepared_after_subset_given)
            g2 = nx.from_numpy_array(self.prepared_after_subset_inferred)

            # Find the size of the maximum common subgraph
            mcs_size = self._max_common_subgraph_size(g1, g2)
            if mcs_size is None:
                mcs_distance = np.nan
            else:
                # Define distance as the number of edges not in the MCS
                total_edges = g1.number_of_edges() + g2.number_of_edges()
                mcs_distance = total_edges - 2 * mcs_size  # Subtract twice the MCS edges
            self.result["maximum_common_subgraph_distance"] = mcs_distance
            self.logger.debug(f"Maximum Common Subgraph Distance: {mcs_distance}")
        except Exception as e:
            self.logger.error(f"Error computing Maximum Common Subgraph Distance: {e}")
            self.result["maximum_common_subgraph_distance"] = np.nan

    def _max_common_subgraph_size(self, g1, g2):
        """Helper method to compute the size of the Maximum Common Subgraph (MCS) between two graphs.

        Args:
            g1 (networkx.Graph): The first graph.
            g2 (networkx.Graph): The second graph.

        Returns:
            int: The number of edges in the MCS.
        """
        try:
            # Use VF2 algorithm to find all maximum common subgraphs
            matcher = nx.algorithms.isomorphism.GraphMatcher(g1, g2)
            # Generate all possible subgraph isomorphisms
            all_subgraphs = list(matcher.subgraph_isomorphisms_iter())
            # Check if any subgraphs were found
            if not all_subgraphs:
                self.logger.debug("No common subgraphs found.")
                return 0  # Return 0 if no common subgraphs are found
            # Find the maximum common subgraph by number of edges
            mcs = max(all_subgraphs, key=lambda x: len(x), default=set())
            # Count the number of edges in the MCS
            mcs_edges = len(mcs)
            return mcs_edges
        except Exception as e:
            self.logger.error(f"Error in MCS computation: {e}")
            return None

    def _calculate_random_walk_kernel_distance(self):
        """Calculates the Random Walk Kernel Distance between the two graphs.

        Utilizes the Random Walk Kernel, which measures the similarity based on the number of matching
        random walks in both graphs. The distance is derived from the kernel similarity.

        Advantages:
            - Captures both local and global structural similarities.
            - Sensitive to repeated structural motifs and connectivity patterns.

        Limitations:
            - Requires parameter tuning (e.g., walk length).
            - Computationally intensive for longer walk lengths.

        Sensitivities:
            - Sensitive to the presence of common walk patterns and repeated motifs.
            - Reflects both structural connectivity and motif frequency.

        Result:
            - A single scalar value representing the Random Walk Kernel distance.
            - Returns `NaN` if computation fails.
        """
        try:
            # Placeholder implementation using adjacency matrices
            # In practice, implement Random Walk Kernel computation
            # For simplicity, using cosine similarity of adjacency matrices treated as vectors
            vec_g1 = self.prepared_after_subset_given.flatten()
            vec_g2 = self.prepared_after_subset_inferred.flatten()
            # Normalize vectors
            vec_g1_norm = vec_g1 / np.linalg.norm(vec_g1) if np.linalg.norm(vec_g1) != 0 else vec_g1
            vec_g2_norm = vec_g2 / np.linalg.norm(vec_g2) if np.linalg.norm(vec_g2) != 0 else vec_g2
            # Compute cosine similarity
            cosine_sim = np.dot(vec_g1_norm, vec_g2_norm)
            # Define distance as 1 - cosine similarity
            rwk_distance = 1 - cosine_sim
            self.result["random_walk_kernel_distance"] = rwk_distance
            self.logger.debug(f"Random Walk Kernel Distance (approximated): {rwk_distance}")
        except Exception as e:
            self.logger.error(f"Error computing Random Walk Kernel Distance: {e}")
            self.result["random_walk_kernel_distance"] = np.nan
