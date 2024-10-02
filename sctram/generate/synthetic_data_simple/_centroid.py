#!/usr/bin/env python3

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import laplacian, shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS, Isomap, SpectralEmbedding
from typing import Optional, Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_complex_adjacency_matrix_with_labels(
    total_nodes: int,
    structures: List[Dict],
    connected_weight_range: Tuple[float, float] = (0.7, 1.0),
    associated_weight_range: Tuple[float, float] = (0.4, 0.6),
    unconnected_weight_range: Tuple[float, float] = (0.0, 0.1),
    inter_structure_connection_prob: float = 0.05,
    seed: Optional[int] = None,
):
    """
    Generate a complex weighted adjacency matrix with specified graph structures, labels, and association-based edge weights.

    Args:
        total_nodes (int): Total number of nodes in the graph.
        structures (list of dict): List of structures to include. Each dict should have:
            - 'name' (str): Unique identifier for the structure.
            - 'type' (str): Type of the structure ('loop', 'linear', 'bifurcation', 'star', 'tree', 'grid', etc.).
            - 'num_nodes' (int): Number of nodes in this structure.
            - 'associated' (list of str): List of names of structures this structure is associated with.
            - Additional parameters depending on the structure type.
        connected_weight_range (tuple, optional): Range for weights of intra-structure edges. Defaults to (0.6, 1.0).
        associated_weight_range (tuple, optional): Range for weights of inter-structure edges between associated structures. Defaults to (0.4, 0.6).
        unconnected_weight_range (tuple, optional): Range for weights of edges not within or between associated structures. Defaults to (0.0, 0.1).
        inter_structure_connection_prob (float, optional): Probability of connecting nodes between associated structures. Defaults to 0.05.
        seed (int, optional): Seed for random number generators for reproducibility. Defaults to None.

    Returns:
        adjacency_matrix (np.ndarray): The generated weighted adjacency matrix with weights between 0 and 1.
        labels (list of str): Labels indicating the structure each node belongs to (e.g., 'loop', 'linear', etc.).
    """
    if seed is not None:
        np.random.seed(seed)

    # Validate input
    total_structure_nodes = sum(structure["num_nodes"] for structure in structures)
    if total_structure_nodes > total_nodes:
        raise ValueError(
            f"Total nodes required by structures ({total_structure_nodes}) exceed total_nodes ({total_nodes})."
        )

    # Initialize the graph
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))  # Nodes are labeled from 0 to total_nodes - 1

    labels = ["other"] * total_nodes  # Initialize all labels as 'other'

    current_node = 0  # Pointer to assign nodes

    # Mapping from structure name to its node indices
    structure_name_to_nodes = {}

    for structure in structures:
        struct_name = structure.get("name")
        struct_type = structure.get("type", "").lower()
        num_nodes = structure.get("num_nodes")

        if not struct_name:
            raise ValueError("Each structure must have a 'name' key.")

        if struct_type not in {"loop", "linear", "bifurcation", "star", "tree", "grid"}:
            raise ValueError(f"Unsupported structure type: {struct_type}")

        if struct_type == "loop":
            if num_nodes < 3:
                raise ValueError("A loop must have at least 3 nodes.")
            subgraph_nodes = list(range(current_node, current_node + num_nodes))
            subg = nx.cycle_graph(n=num_nodes)
            mapping = {i: node for i, node in enumerate(subgraph_nodes)}
            subg = nx.relabel_nodes(subg, mapping)
            G.add_edges_from(subg.edges())
            for node in subgraph_nodes:
                labels[node] = "loop"
            structure_name_to_nodes[struct_name] = subgraph_nodes
            current_node += num_nodes

        elif struct_type == "linear":
            if num_nodes < 2:
                raise ValueError("A linear structure must have at least 2 nodes.")
            subgraph_nodes = list(range(current_node, current_node + num_nodes))
            subg = nx.path_graph(n=num_nodes)
            mapping = {i: node for i, node in enumerate(subgraph_nodes)}
            subg = nx.relabel_nodes(subg, mapping)
            G.add_edges_from(subg.edges())
            for node in subgraph_nodes:
                labels[node] = "linear"
            structure_name_to_nodes[struct_name] = subgraph_nodes
            current_node += num_nodes

        elif struct_type == "bifurcation":
            # Bifurcation: one root node branching into multiple paths
            branches = structure.get("branches", 2)  # Default to 2 branches
            min_nodes = 1 + branches  # At least 1 root + 1 per branch
            if num_nodes < min_nodes:
                raise ValueError(f"A bifurcation must have at least {min_nodes} nodes for {branches} branches.")
            subgraph_nodes = list(range(current_node, current_node + num_nodes))
            root = subgraph_nodes[0]
            G.add_node(root)
            labels[root] = "bifurcation_root"
            nodes_per_branch = (num_nodes - 1) // branches
            for b in range(branches):
                start = 1 + b * nodes_per_branch
                end = start + nodes_per_branch
                if b == branches - 1:
                    # Last branch takes the remaining nodes
                    end = num_nodes
                branch_nodes = subgraph_nodes[start:end]
                if len(branch_nodes) == 0:
                    continue
                G.add_edge(root, branch_nodes[0])
                for i in range(len(branch_nodes) - 1):
                    G.add_edge(branch_nodes[i], branch_nodes[i + 1])
                for node in branch_nodes:
                    labels[node] = "bifurcation_branch"
            structure_name_to_nodes[struct_name] = subgraph_nodes
            current_node += num_nodes

        elif struct_type == "star":
            if num_nodes < 2:
                raise ValueError("A star structure must have at least 2 nodes.")
            subgraph_nodes = list(range(current_node, current_node + num_nodes))
            center = subgraph_nodes[0]
            leaves = subgraph_nodes[1:]
            for leaf in leaves:
                G.add_edge(center, leaf)
            labels[center] = "star_center"
            for leaf in leaves:
                labels[leaf] = "star_leaf"
            structure_name_to_nodes[struct_name] = subgraph_nodes
            current_node += num_nodes

        elif struct_type == "tree":
            # Simple balanced binary tree
            depth = structure.get("depth", 3)
            expected_nodes = sum([2**i for i in range(depth + 1)])
            if num_nodes < expected_nodes:
                raise ValueError(f"A balanced binary tree of depth {depth} requires at least {expected_nodes} nodes.")
            subgraph_nodes = list(range(current_node, current_node + num_nodes))
            subg = nx.balanced_tree(r=2, h=depth)
            if subg.number_of_nodes() > num_nodes:
                raise ValueError(f"Binary tree with depth {depth} exceeds the number of nodes specified ({num_nodes}).")
            mapping = {i: node for i, node in enumerate(subgraph_nodes)}
            subg = nx.relabel_nodes(subg, mapping)
            G.add_edges_from(subg.edges())
            for node in subgraph_nodes:
                labels[node] = "tree"
            structure_name_to_nodes[struct_name] = subgraph_nodes
            current_node += num_nodes

        elif struct_type == "grid":
            # Grid structure (2D grid)
            rows = structure.get("rows", 3)
            cols = structure.get("cols", 3)
            required_nodes = rows * cols
            if num_nodes != required_nodes:
                raise ValueError(f"For grid structure, num_nodes must equal rows * cols ({rows * cols}).")
            subgraph_nodes = list(range(current_node, current_node + num_nodes))
            subg = nx.grid_2d_graph(rows, cols)
            mapping = {(i, j): subgraph_nodes[i * cols + j] for i, j in subg.nodes()}
            subg = nx.relabel_nodes(subg, mapping)
            G.add_edges_from(subg.edges())
            for node in subgraph_nodes:
                labels[node] = "grid"
            structure_name_to_nodes[struct_name] = subgraph_nodes
            current_node += num_nodes

        else:
            raise ValueError(f"Unsupported structure type: {struct_type}")

    # Initialize the adjacency matrix with unconnected weights
    adjacency_matrix = np.random.uniform(
        low=unconnected_weight_range[0], high=unconnected_weight_range[1], size=(total_nodes, total_nodes)
    )
    # Make the matrix symmetric by copying the upper triangle to the lower triangle
    adjacency_matrix = np.triu(adjacency_matrix)  # Upper triangle
    adjacency_matrix += adjacency_matrix.T - np.diag(adjacency_matrix.diagonal())  # Mirror upper to lower

    # Assign weights to intra-structure edges
    for u, v in G.edges():
        weight = np.random.uniform(low=connected_weight_range[0], high=connected_weight_range[1])
        adjacency_matrix[u, v] = weight
        adjacency_matrix[v, u] = weight  # Ensure symmetry

    # Assign weights to inter-structure edges based on associations
    processed_pairs = set()  # To avoid processing the same pair twice
    for structure in structures:
        struct_name = structure['name']
        associated_structures = structure.get('associated', [])
        struct_nodes = structure_name_to_nodes.get(struct_name, [])

        for assoc_struct_name in associated_structures:
            # Avoid processing the same pair twice
            pair = tuple(sorted([struct_name, assoc_struct_name]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            assoc_struct_nodes = structure_name_to_nodes.get(assoc_struct_name, [])
            if not assoc_struct_nodes:
                continue  # No nodes in the associated structure

            # Connect nodes between struct_nodes and assoc_struct_nodes based on inter_structure_connection_prob
            for u in struct_nodes:
                for v in assoc_struct_nodes:
                    if u == v:
                        continue  # Skip self-loop
                    if np.random.rand() < inter_structure_connection_prob:
                        weight = np.random.uniform(low=associated_weight_range[0], high=associated_weight_range[1])
                        adjacency_matrix[u, v] = weight
                        adjacency_matrix[v, u] = weight  # Ensure symmetry

    np.fill_diagonal(adjacency_matrix, 0.0)  # No self-loops

    return adjacency_matrix, labels


def generate_random_adjacency_matrix(graph_type="erdos_renyi", num_nodes=100, weighted=True, **kwargs):
    """
    Generate a complex adjacency matrix for testing purposes with weights between 0 and 1.

    Args:
        graph_type (str): The type of graph to generate. Options are:
            - 'erdos_renyi': Random graph
            - 'barabasi_albert': Scale-free network
            - 'watts_strogatz': Small-world network
            - 'stochastic_block': Graph with community structure
            - 'random_regular': Regular graph
            - 'powerlaw_cluster': Power law cluster graph
            - 'custom': Provide your own NetworkX graph via kwargs['graph']
        num_nodes (int): Number of nodes in the graph
        weighted (bool): If True, assign random weights between 0 and 1 to edges.
        **kwargs: Additional keyword arguments specific to the graph type.

    Returns:
        np.ndarray: Adjacency matrix of the generated graph with weights between 0 and 1
    """
    if graph_type == "erdos_renyi":
        p = kwargs.get("p", 0.05)  # Probability for edge creation
        g = nx.erdos_renyi_graph(n=num_nodes, p=p, seed=kwargs.get("seed", None))
    elif graph_type == "barabasi_albert":
        m = kwargs.get("m", max(1, num_nodes // 100))  # Number of edges to attach from a new node
        g = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=kwargs.get("seed", None))
    elif graph_type == "watts_strogatz":
        k = kwargs.get("k", max(2, num_nodes // 10))  # Each node is connected to k nearest neighbors in ring topology
        p = kwargs.get("p", 0.1)  # Probability of rewiring each edge
        g = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p, seed=kwargs.get("seed", None))
    elif graph_type == "stochastic_block":
        sizes = kwargs.get("sizes", [num_nodes // 3] * 3)
        probs = kwargs.get("probs", [[0.1, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.1]])
        g = nx.stochastic_block_model(sizes, probs, seed=kwargs.get("seed", None))
    elif graph_type == "random_regular":
        d = kwargs.get("d", max(2, num_nodes // 10))  # Degree of each node
        if d * num_nodes % 2 != 0:
            raise ValueError("For random_regular_graph, d*num_nodes must be even.")
        g = nx.random_regular_graph(d=d, n=num_nodes, seed=kwargs.get("seed", None))
    elif graph_type == "powerlaw_cluster":
        m = kwargs.get("m", max(1, num_nodes // 100))  # Number of edges to attach from a new node
        p = kwargs.get("p", 0.1)  # Probability of adding a triangle after adding a random edge
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=m, p=p, seed=kwargs.get("seed", None))
    elif graph_type == "custom":
        g = kwargs.get("graph", None)
        if g is None:
            raise ValueError("For 'custom' graph type, provide a NetworkX graph via 'graph' keyword argument.")
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    if weighted:
        # Assign random weights between 0 and 1 to each edge
        for u, v in g.edges():
            g[u][v]["weight"] = np.random.uniform(0.0, 1.0)
    else:
        # For unweighted graphs, ensure all existing edges have weight 1.0
        for u, v in g.edges():
            g[u][v]["weight"] = 1.0

    # Convert to adjacency matrix with weights
    if weighted:
        # Use the 'weight' attribute for edge weights
        adjacency_matrix = nx.to_numpy_array(g, weight="weight")
    else:
        # If unweighted, use the 'weight' attribute set to 1.0
        adjacency_matrix = nx.to_numpy_array(g, weight=None)
        adjacency_matrix[adjacency_matrix != 0] = 1.0  # Ensure binary weights

    # Ensure the adjacency matrix is of type float and values are between 0 and 1
    adjacency_matrix = np.array(adjacency_matrix, dtype=float)
    adjacency_matrix = np.clip(adjacency_matrix, 0.0, 1.0)

    return adjacency_matrix


class LaplacianEigenmaps:
    def __init__(self, n_components):
        """Initializes the Laplacian Eigenmaps model.

        Args:
            n_components (int): The number of dimensions for the reduced space.
        """
        self.n_components = n_components

    def fit_transform(self, X):
        """Compute the low-dimensional embedding."""
        # Compute the normalized Laplacian matrix
        laplacian_matrix = laplacian(X, normed=True)
        # Compute the first `n_components + 1` smallest eigenvalues and eigenvectors
        _, eigvecs = eigsh(laplacian_matrix, k=self.n_components + 1, which="SM")
        # Discard the first eigenvector (corresponding to the zero eigenvalue)
        embedding = eigvecs[:, 1 : self.n_components + 1]
        return embedding


class CentroidCalculator:
    """Calculates centroids using various dimensionality reduction methods."""

    def __init__(
        self, centroid_method: str, adjacency_matrix: np.ndarray, num_features: int, random_state: int
    ) -> None:
        """Initializes the CentroidCalculator.

        Args:
            centroid_method (str): The method used for centroid calculation. Choices are 'mds', 'spectral', 'isomap', 'kernel_pca', and 'laplacian'.
            adjacency_matrix (np.ndarray): The adjacency matrix representing the graph (or any square matrix).
            num_features (int): The number of dimensions for the reduced space.
            random_state (int): A seed value to ensure reproducibility.
        """
        self.centroid_method = centroid_method
        self.adjacency_matrix = adjacency_matrix.copy()
        self.num_features = num_features
        self.random_state = random_state

        if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise ValueError("The adjacency matrix must be a square matrix.")

        if not np.all(np.diag(self.adjacency_matrix) == 0):
            raise ValueError("All diagonal elements of the adjacency matrix must be zero.")

        if not np.allclose(self.adjacency_matrix, self.adjacency_matrix.T):
            raise ValueError("Adjacency matrix must be symmetrical.")

    def get(self):
        """Create k-dimensional centroids using the specified method based on the edge weights.

        Each method has specific characteristics and use cases:

        'mds': Multidimensional Scaling (MDS)
            - Preserves the pairwise distance between nodes as well as possible in the lower-dimensional space.
            - Useful for visualizing similarity or dissimilarity data, such as in social networks where edges represent similarity in attributes.

        'spectral': Spectral Embedding
            - Utilizes the graph Laplacian to perform a low-dimensional embedding, emphasizing the clustering of nodes in the graph based on their connectivity.
            - Best used for graphs where you expect to find a clear clustering structure or communities, such as in citation networks or community detection tasks.

        'isomap': Isometric Mapping
            - Seeks to preserve the geodesic distances in the original high-dimensional space. It is effectively an MDS that uses geodesic distances instead of Euclidean ones.
            - Ideal for data where global manifold structure is important, such as in data that lie on a curved manifold, offering a perspective that maintains global properties.

        'kernel_pca': Kernel PCA
            - Extends PCA to nonlinear dimensionality reduction through the use of kernels, capable of capturing complex structures in the data.
            - Suitable for data where nonlinear relationships are significant, such as in image processing or any complex network structures where linear projections fail to capture the essence of data relations.

        'laplacian': Laplacian Eigenmaps
            - Focuses on preserving local node relationships and is particularly sensitive to the choice of neighborhood or the graph construction.
            - Effective for data where local relationships are more significant than global relationships, such as in spectral clustering or when data has a very irregular structure.

        Returns:
            np.ndarray: The centroids in the reduced space, capturing essential relationships as specified by the method.
        """
        if self.centroid_method == "mds":
            return self._create_centroids_mds()
        elif self.centroid_method == "spectral":
            return self._create_centroids_spectral()
        elif self.centroid_method == "isomap":
            return self._create_centroids_isomap()
        elif self.centroid_method == "kernel_pca":
            return self._create_centroids_kernel_pca()
        elif self.centroid_method == "laplacian":
            return self._create_centroids_laplacian()
        else:
            raise ValueError(f"Unknown centroid method: {self.centroid_method}")

    def _compute_distance_matrix(self):
        """Compute the distance matrix from the adjacency (similarity) matrix."""
        # Convert similarities to distances: higher similarities correspond to shorter distances
        epsilon = 1e-8  # Small value to prevent division by zero
        similarity_matrix = self.adjacency_matrix + epsilon
        # Invert similarities to get distances
        distance_matrix = 1 / similarity_matrix
        # Now compute the shortest path distances
        distance_matrix = shortest_path(csgraph=distance_matrix, method="D", directed=False, unweighted=False)
        return distance_matrix

    def _create_centroids_mds(self) -> np.ndarray:
        """Multidimensional Scaling (MDS)."""
        distance_matrix = self._compute_distance_matrix()
        mds = MDS(
            n_components=self.num_features,
            dissimilarity="precomputed",
            random_state=self.random_state,
        )
        centroids = mds.fit_transform(distance_matrix)
        return centroids

    def _create_centroids_spectral(self) -> np.ndarray:
        """Spectral Embedding."""
        embedding = SpectralEmbedding(
            n_components=self.num_features,
            affinity="precomputed",
            random_state=self.random_state,
        )
        centroids = embedding.fit_transform(self.adjacency_matrix)
        return centroids

    def _create_centroids_isomap(self) -> np.ndarray:
        """Isomap."""
        distance_matrix = self._compute_distance_matrix()
        isomap = Isomap(n_components=self.num_features, metric="precomputed")
        centroids = isomap.fit_transform(distance_matrix)
        return centroids

    def _create_centroids_kernel_pca(self, gamma=None) -> np.ndarray:
        """Kernel PCA."""
        # Compute the kernel matrix using RBF kernel based on distances
        distance_matrix = self._compute_distance_matrix()
        if gamma is None:
            # Set gamma as the inverse of the median of the distances (excluding infinities)
            finite_distances = distance_matrix[np.isfinite(distance_matrix)]
            gamma = 1 / np.median(finite_distances)
        # Compute RBF kernel from distances
        kernel_matrix = np.exp(-gamma * distance_matrix**2)
        kernel_pca = KernelPCA(
            n_components=self.num_features,
            kernel="precomputed",
            random_state=self.random_state,
        )
        centroids = kernel_pca.fit_transform(kernel_matrix)
        return centroids

    def _create_centroids_laplacian(self) -> np.ndarray:
        """Laplacian Eigenmaps."""
        laplacian_eigenmaps = LaplacianEigenmaps(n_components=self.num_features)
        centroids = laplacian_eigenmaps.fit_transform(self.adjacency_matrix)
        return centroids


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class DataGenerator:
    """Generates a dataset by populating around centroids and along edges between centroids."""
    
    def __init__(
        self,
        centroids: np.ndarray,
        adjacency_matrix: np.ndarray,
        labels: np.ndarray,
        total_points: int,
        centroid_proportion: float = 0.7,
        cluster_distribution: str = 'gaussian',
        cluster_distribution_z: float = 1.96,
        edge_distribution: str = 'reverse_gaussian',
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
            centroid_proportion (float, optional): Proportion of points around centroids. Defaults to 0.7.
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
        
        valid_distributions = ['gaussian', 'uniform']
        if self.cluster_distribution not in valid_distributions:
            raise ValueError(f"Unsupported cluster_distribution: {self.cluster_distribution}. Supported distributions: {valid_distributions}")
        
        valid_edge_distributions = ['uniform', 'beta', 'cosine', 'gaussian', 'reverse_gaussian', 'triangular']
        if self.edge_distribution not in valid_edge_distributions:
            raise ValueError(f"Unsupported edge_distribution: {self.edge_distribution}. Supported distributions: {valid_edge_distributions}")
    
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
        data_points_centroids = self._generate_points_around_centroids(
            num_points=num_points_centroids
        )
        
        # Generate points along edges
        data_points_edges = self._generate_points_between_centroids(
            num_points=num_points_edges
        )
        
        # Combine data points
        data_points = pd.concat([data_points_centroids, data_points_edges], ignore_index=True)
        
        # Ensure exact number of total points
        data_points = self._adjust_total_points(data_points)
        
        # Add centroid information
        centroid_information = pd.DataFrame(
            np.full(
                shape=(num_centroids, data_points.shape[1] - dimensionality),
                fill_value=None
            ),
            columns=data_points.columns[dimensionality:]
        )
        centroid_information["type"] = "centroids"
        centroid_information["node_index"] = list(range(len(self.labels)))
        centroid_information["node_label"] = self.labels.copy()
        centroid_information["node_index_extended"] = centroid_information["node_index"]
        centroids_df = pd.DataFrame(self.centroids, columns=[f'feature_{i}' for i in range(dimensionality)])
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
            if self.cluster_distribution == 'gaussian':
                points = np.random.normal(
                    loc=centroid,
                    scale=self.cluster_std,
                    size=(n_points, dimensionality)
                )
            elif self.cluster_distribution == 'uniform':
                # The range is set to achieve the desired standard deviation
                range_width = self.cluster_std * 2 * np.sqrt(3)
                lower = centroid - range_width / 2
                upper = centroid + range_width / 2
                points = np.random.uniform(
                    low=lower,
                    high=upper,
                    size=(n_points, dimensionality)
                )
            else:
                raise ValueError(f"Unsupported cluster_distribution: {self.cluster_distribution}")
            df = pd.DataFrame(points, columns=[f'feature_{i}' for i in range(dimensionality)])
            df['type'] = 'cluster'
            df['node_index'] = idx
            df['node_label'] = label
            df['edge_indices'] = np.nan
            df['edge_labels'] = np.nan
            df['node_index_extended'] = idx
            data_frames.append(df)
        
        if data_frames:
            data_points = pd.concat(data_frames, ignore_index=True)
        else:
            # No cluster points generated
            data_points = pd.DataFrame(columns=[f'feature_{i}' for i in range(dimensionality)] + 
                                                   ['type', 'node_index', 'node_label', 'edge_indices', 'edge_labels', 'node_index_extended'])
        
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
            return pd.DataFrame(columns=[f'feature_{i}' for i in range(dimensionality)] + 
                                         ['type', 'node_index', 'node_label', 'edge_indices', 'edge_labels', 'node_index_extended'])
        
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
        for (edge, num_edge_points) in zip(edges, points_per_edge):
            i, j, weight = edge
            centroid_i = self.centroids[i]
            centroid_j = self.centroids[j]
            label_i = self.labels[i]
            label_j = self.labels[j]
            if num_edge_points <= 0:
                continue  # Skip if no points to generate
            
            # Generate coefficients based on edge_distribution
            coefficients = self._generate_edge_coefficients(self.edge_distribution, self.edge_distribution_params, num_edge_points)
            
            # Linear interpolation
            points = centroid_i + coefficients * (centroid_j - centroid_i)
            
            # Add noise
            noise = np.random.normal(
                loc=0,
                scale=self.edge_std,
                size=(num_edge_points, dimensionality)
            )
            points += noise
        
            df = pd.DataFrame(points, columns=[f'feature_{k}' for k in range(dimensionality)])
            df['type'] = 'edge'
            df['node_index'] = np.nan
            df['node_label'] = np.nan
            df['edge_indices'] = f'{i}-{j}'
            df['edge_labels'] = f'{label_i}-{label_j}'
            # Compute node_index_extended
            distances = cdist(points, self.centroids)
            closest_centroids = np.argmin(distances, axis=1)
            df['node_index_extended'] = closest_centroids
            data_frames.append(df)
        
        if data_frames:
            data_points = pd.concat(data_frames, ignore_index=True)
        else:
            # No edge points generated
            data_points = pd.DataFrame(columns=[f'feature_{i}' for i in range(dimensionality)] + 
                                               ['type', 'node_index', 'node_label', 'edge_indices', 'edge_labels', 'node_index_extended'])
        
        return data_points
    
    @staticmethod
    def _generate_edge_coefficients(edge_distribution: str, edge_distribution_params: dict, num_edge_points: int) -> np.ndarray:
        """
        Generates interpolation coefficients for edge points based on the specified distribution.
        
        Args:
            num_edge_points (int): Number of points to generate.
        
        Returns:
            np.ndarray: Coefficients for interpolation.
        """        
        if edge_distribution == 'uniform':
            coefficients = np.random.uniform(0, 1, size=(num_edge_points, 1))
        elif edge_distribution == 'gaussian':
            # Normal distribution centered at 0.5
            mu = edge_distribution_params.get('mu', 0.5)
            sigma = edge_distribution_params.get('sigma', 0.15)
            coefficients = np.random.normal(loc=mu, scale=sigma, size=(num_edge_points, 1))
            coefficients = np.clip(coefficients, 0, 1)
        elif edge_distribution == 'beta':
            # Default parameters for a smoother distribution
            a = edge_distribution_params.get('a', 2)
            b = edge_distribution_params.get('b', 2)
            coefficients = np.random.beta(a=a, b=b, size=(num_edge_points, 1))
        elif edge_distribution == 'reverse_gaussian':
            # U-shaped distribution using 1 - Gaussian PDF
            sigma = edge_distribution_params.get('sigma', 0.15)
            x = np.linspace(0, 1, num_edge_points)
            gaussian_pdf = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
            probabilities = 1 - gaussian_pdf / gaussian_pdf.max()
            probabilities /= probabilities.sum()
            coefficients = np.random.choice(x, size=num_edge_points, p=probabilities)
            coefficients = coefficients.reshape(-1, 1)
        elif edge_distribution == 'cosine':
            # U-shaped distribution using cosine function
            power = edge_distribution_params.get('power', 1)
            x = np.linspace(0, 1, num_edge_points)
            cosine_values = 0.5 * (1 + np.cos(np.pi * x))
            probabilities = cosine_values ** power
            probabilities /= probabilities.sum()
            coefficients = np.random.choice(x, size=num_edge_points, p=probabilities)
            coefficients = coefficients.reshape(-1, 1)
        elif edge_distribution == 'triangular':
            # U-shaped triangular distribution similar to reverse_gaussian
            # Combine two symmetric triangular distributions: one peaked at 0, another at 1
            # Introduce sharpness parameter (default to 0.5 for a balanced distribution)
            sharpness = edge_distribution_params.get('sharpness', 0.5)
            sharpness = np.clip(sharpness, 0.0, 1.0) # Ensure sharpness is within (0, 1)
            
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
