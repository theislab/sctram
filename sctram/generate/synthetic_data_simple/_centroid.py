#!/usr/bin/env python3

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import laplacian, shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS, Isomap, SpectralEmbedding
from typing import Optional


def generate_complex_adjacency_matrix_with_labels(
    total_nodes: int,
    structures: list,
    connected_weight_range: tuple = (0.6, 1.0),
    unconnected_weight_range: tuple = (0.0, 0.4),
    seed: Optional[int] = None,
):
    """
    Generate a complex weighted adjacency matrix with specified graph structures and labels.

    Args:
        total_nodes (int): Total number of nodes in the graph.
        structures (list of dict): List of structures to include. Each dict should have:
            - 'type' (str): Type of the structure ('loop', 'linear', 'bifurcation', 'star', 'tree', 'grid', etc.).
            - 'num_nodes' (int): Number of nodes in this structure.
            - Additional parameters depending on the structure type.
        connected_weight_range (tuple, optional): Range for weights of connected nodes. Defaults to (0.6, 1.0).
        unconnected_weight_range (tuple, optional): Range for weights of unconnected nodes. Defaults to (0.0, 0.4).
        seed (int, optional): Seed for random number generators for reproducibility. Defaults to None.

    Returns:
        adjacency_matrix (np.ndarray): The generated weighted adjacency matrix with weights between 0 and 1.
        labels (list of str): Labels indicating the structure each node belongs to ('loop', 'linear', etc.).
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

    for structure in structures:
        struct_type = structure["type"].lower()
        num_nodes = structure["num_nodes"]

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

    np.fill_diagonal(adjacency_matrix, 0.0)  # No self-loops
    # Assign weights to connected edges
    for u, v in G.edges():
        weight = np.random.uniform(low=connected_weight_range[0], high=connected_weight_range[1])
        adjacency_matrix[u, v] = weight
        adjacency_matrix[v, u] = weight  # Ensure symmetry

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


class DataGenerator:
    """Generates a dataset by populating around centroids and along edges between centroids."""

    def __init__(
        self,
        centroids: np.ndarray,
        adjacency_matrix: np.ndarray,
        total_points: int,
        centroid_proportion: float = 0.7,
        cluster_std: float = 0.05,
        edge_std: float = 0.01,
        random_state: int = None,
    ):
        """
        Initializes the DataGenerator.

        Args:
            centroids (np.ndarray): Centroid coordinates.
            adjacency_matrix (np.ndarray): Adjacency matrix with edge weights.
            total_points (int): Total number of data points to generate.
            centroid_proportion (float, optional): Proportion of points around centroids. Defaults to 0.7.
            cluster_std (float, optional): Std deviation for clusters. Defaults to 0.05.
            edge_std (float, optional): Std deviation for edges. Defaults to 0.01.
            random_state (int, optional): Seed for reproducibility. Defaults to None.
        """
        self.centroids = centroids
        self.adjacency_matrix = adjacency_matrix
        self.total_points = total_points
        self.centroid_proportion = centroid_proportion
        self.cluster_std = cluster_std
        self.edge_std = edge_std
        self.random_state = random_state

        # Validate inputs
        self._validate_inputs()

        # Set random state for reproducibility
        self._set_random_state()

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
        if self.total_points <= 0:
            raise ValueError("total_points must be a positive integer.")
        if self.cluster_std <= 0:
            raise ValueError("cluster_std must be positive.")
        if self.edge_std <= 0:
            raise ValueError("edge_std must be positive.")

    def _set_random_state(self):
        """Sets the random state for reproducibility."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def generate_data(self):
        """
        Generates the dataset based on the centroids and adjacency matrix.

        Returns:
            np.ndarray: Generated data points (total_points x dimensionality).
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
        data_points = np.vstack([data_points_centroids, data_points_edges])

        # Ensure exact number of total points
        data_points = self._adjust_total_points(data_points)

        return data_points

    def _generate_points_around_centroids(self, num_points: int) -> np.ndarray:
        """
        Generates data points around each centroid using a Gaussian distribution.

        Args:
            num_points (int): Total number of points to generate.

        Returns:
            np.ndarray: Generated data points.
        """
        num_centroids, dimensionality = self.centroids.shape
        points_per_centroid = num_points // num_centroids
        remainder = num_points % num_centroids
        data_points = []

        for idx, centroid in enumerate(self.centroids):
            n_points = points_per_centroid + (1 if idx < remainder else 0)
            points = np.random.normal(loc=centroid, scale=self.cluster_std, size=(n_points, dimensionality))
            data_points.append(points)

        data_points = np.vstack(data_points)
        return data_points

    def _generate_points_between_centroids(self, num_points: int) -> np.ndarray:
        """
        Generates data points along the edges between centroids, with density proportional to edge weights.

        Args:
            num_points (int): Total number of points to generate.

        Returns:
            np.ndarray: Generated data points.
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
            return np.empty((0, dimensionality))

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
                    points_per_edge[idx % len(points_per_edge)] -= 1

        data_points = []
        for edge, num_edge_points in zip(edges, points_per_edge):
            i, j, weight = edge
            centroid_i = self.centroids[i]
            centroid_j = self.centroids[j]
            if num_edge_points <= 0:
                continue  # Skip if no points to generate
            # Linear interpolation coefficients
            coefficients = np.random.uniform(0, 1, size=(num_edge_points, 1))
            points = centroid_i + coefficients * (centroid_j - centroid_i)
            # Add Gaussian noise
            noise = np.random.normal(loc=0, scale=self.edge_std, size=(num_edge_points, dimensionality))
            points += noise
            data_points.append(points)

        if data_points:
            data_points = np.vstack(data_points)
        else:
            data_points = np.empty((0, dimensionality))

        return data_points

    def _adjust_total_points(self, data_points: np.ndarray) -> np.ndarray:
        """
        Adjusts the number of data points to exactly match total_points.

        Args:
            data_points (np.ndarray): Combined data points.

        Returns:
            np.ndarray: Adjusted data points.
        """
        num_generated_points = data_points.shape[0]
        if num_generated_points > self.total_points:
            # Randomly select total_points data points
            indices = np.random.choice(num_generated_points, self.total_points, replace=False)
            data_points = data_points[indices]
        elif num_generated_points < self.total_points:
            # Generate additional points around centroids
            num_missing = self.total_points - num_generated_points
            extra_points = self._generate_points_around_centroids(num_missing)
            data_points = np.vstack([data_points, extra_points])
        # Final check
        assert data_points.shape[0] == self.total_points
        return data_points
