#!/usr/bin/env python3

from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


class DiffmapAdjacencyConverter:
    """A class to convert Diffusion Map embeddings into an adjacency matrix using various methods.

    The Diffusion Map embedding represents cells in a lower-dimensional diffusion space,
    capturing the manifold structure of the data based on diffusion processes.

    Converting Diffmap embeddings into an adjacency matrix enables the construction
    of graph-based representations of cellular relationships in diffusion space, facilitating
    downstream analyses such as trajectory inference, clustering, and network analysis.

    Supported Methods:
        - k-Nearest Neighbors (k-NN) based adjacency
        - Threshold-based adjacency
        - Minimum Spanning Tree (MST) based adjacency
        - Gaussian Kernel-Based Adjacency

    Attributes:
        diffmap_embedding (np.ndarray): Array of Diffusion Map embeddings for each cell (n_cells x n_components).
        adjacency_matrix (Optional[np.ndarray]): The resulting adjacency matrix after conversion.
        cell_labels (Optional[np.ndarray]): Array of labels for each cell (n_cells,).
    """

    def __init__(self, diffmap_embedding: np.ndarray, cell_labels: Optional[np.ndarray] = None):
        """Initializes the DiffmapAdjacencyConverter with a Diffusion Map embedding array.

        Args:
            diffmap_embedding (np.ndarray): A two-dimensional array of Diffusion Map embeddings (n_cells x n_components).
            cell_labels (Optional[np.ndarray]): An optional array of labels for each cell.

        Raises:
            ValueError: If diffmap_embedding is not a two-dimensional numpy array.
            TypeError: If `diffmap_embedding` is not a numpy array.
        """
        if not isinstance(diffmap_embedding, np.ndarray):
            raise TypeError("diffmap_embedding must be a numpy array.")
        if diffmap_embedding.ndim != 2:
            raise ValueError("diffmap_embedding must be a two-dimensional array of shape (n_cells, n_components).")

        self.diffmap_embedding = diffmap_embedding
        self.adjacency_matrix: Optional[np.ndarray] = None

        if cell_labels is not None:
            if not isinstance(cell_labels, np.ndarray):
                raise TypeError("cell_labels must be a numpy array.")
            if cell_labels.ndim != 1:
                raise ValueError("cell_labels must be a one-dimensional array.")
            if len(cell_labels) != diffmap_embedding.shape[0]:
                raise ValueError("cell_labels must have the same length as the number of cells in diffmap_embedding.")
            self.cell_labels = cell_labels
        else:
            self.cell_labels = None

    def _handle_disconnected_components(self, adjacency: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """Handles disconnected components in the adjacency matrix by connecting them with epsilon-weighted edges.

        Args:
            adjacency (np.ndarray): The initial adjacency matrix.
            epsilon (float): A small value to assign to the connecting edges. Default is 1e-5.

        Returns:
            np.ndarray: An adjacency matrix with connected components connected via epsilon edges.
        """
        graph = nx.from_numpy_array(adjacency)
        if nx.is_connected(graph):
            return adjacency

        # Find connected components
        components = list(nx.connected_components(graph))
        num_components = len(components)

        # Connect all components in a linear chain with epsilon-weighted edges
        for i in range(num_components - 1):
            comp_a = list(components[i])
            comp_b = list(components[i + 1])
            node_a = comp_a[0]
            node_b = comp_b[0]
            adjacency[node_a, node_b] = epsilon
            adjacency[node_b, node_a] = epsilon

        return adjacency

    def to_knn_adjacency(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        include_self: bool = False,
        weighted: bool = False,
        handle_disconnected: bool = True,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Converts the Diffusion Map embedding into an adjacency matrix using the k-Nearest Neighbors (k-NN) method.

        In this method, each cell is connected to its k nearest neighbors based on the distances in diffusion space.

        Mathematical Formulation:
            1.  Let x_i be the Diffusion Map embedding of cell i.
            2.  Compute the distance between cells using the specified metric in diffusion space.
            3.  For each cell i, identify the k cells with the smallest distances to i.
            4.  Set A_{i,j} = distance(i, j) if weighted is True, else A_{i,j} = 1 if j
                is among the k nearest neighbors of i; otherwise, A_{i,j} = 0.
            5.  Ensure the adjacency matrix is symmetric.
            6.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            n_neighbors (int): Number of nearest neighbors to connect. Must be a positive integer. Default is 5.
            metric (str): Distance metric to use. Default is 'euclidean'.
            algorithm (str): Algorithm to compute nearest neighbors. Default is 'auto'.
            include_self (bool): Whether to include the cell itself as its neighbor. Default is False.
            weighted (bool): Whether to assign edge weights based on distance.
                If False, edges are binary. Default is False.
            handle_disconnected (bool): Whether to handle disconnected components by
                connecting them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when handling
                disconnected components. Default is 1e-5.

        Returns:
            np.ndarray: An adjacency matrix of shape (n_cells, n_cells).

        Raises:
            ValueError: If `n_neighbors` is not a positive integer.
        """
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")

        # Use NearestNeighbors from sklearn
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + int(include_self), metric=metric, algorithm=algorithm).fit(
            self.diffmap_embedding
        )

        distances, indices = nbrs.kneighbors(self.diffmap_embedding)

        n_cells = self.diffmap_embedding.shape[0]
        adjacency = np.zeros((n_cells, n_cells), dtype=float if weighted else int)

        for idx, neighbors in enumerate(indices):
            start = 1 if not include_self else 0
            for neighbor_idx in range(start, len(neighbors)):
                neighbor = neighbors[neighbor_idx]
                if weighted:
                    adjacency[idx, neighbor] = distances[idx, neighbor_idx]
                else:
                    adjacency[idx, neighbor] = 1
                # Ensure symmetry
                if weighted:
                    adjacency[neighbor, idx] = distances[idx, neighbor_idx]
                else:
                    adjacency[neighbor, idx] = 1

        if handle_disconnected:
            adjacency = self._handle_disconnected_components(adjacency, epsilon=epsilon)

        self.adjacency_matrix = adjacency
        return self.adjacency_matrix

    def to_threshold_adjacency(
        self,
        delta: float,
        metric: str = "euclidean",
        weighted: bool = False,
        handle_disconnected: bool = True,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Converts the Diffusion Map embedding into an adjacency matrix by thresholding distances in diffusion space.

        In this method, two cells are connected if the distance between them in diffusion space
        is less than a specified threshold delta.

        Mathematical Formulation:
            1.  Let x_i and x_j be the Diffusion Map embeddings of cells i and j, respectively.
            2.  Compute the distance d(i, j) = metric(x_i, x_j).
            3.  Define A_{i,j} = d(i, j) if weighted is True and d(i, j) < delta;
                otherwise, A_{i,j} = 1 if d(i, j) < delta; else A_{i,j} = 0.
            4.  Ensure the adjacency matrix is symmetric.
            5.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            delta (float): The maximum allowed distance for two cells to be connected. Must be a positive number.
            metric (str): Distance metric to use. Default is 'euclidean'.
            weighted (bool): Whether to assign edge weights based on distance. If False, edges are binary. Default is False.
            handle_disconnected (bool): Whether to handle disconnected components by
                connecting them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when handling disconnected components. Default is 1e-5.

        Returns:
            np.ndarray: An adjacency matrix of shape (n_cells, n_cells).

        Raises:
            ValueError: If `delta` is not a positive number.
        """
        if not isinstance(delta, (int, float)) or delta <= 0:
            raise ValueError("delta must be a positive number.")

        # Compute pairwise distances
        pairwise_dist = squareform(pdist(self.diffmap_embedding, metric=metric))

        n_cells = self.diffmap_embedding.shape[0]
        adjacency = np.zeros((n_cells, n_cells), dtype=float if weighted else int)

        if weighted:
            adjacency[pairwise_dist < delta] = pairwise_dist[pairwise_dist < delta]
        else:
            adjacency[pairwise_dist < delta] = 1

        # Remove self-connections
        np.fill_diagonal(adjacency, 0)

        if handle_disconnected:
            adjacency = self._handle_disconnected_components(adjacency, epsilon=epsilon)

        self.adjacency_matrix = adjacency
        return self.adjacency_matrix

    def to_mst_adjacency(
        self,
        metric: str = "euclidean",
        algorithm: str = "kruskal",
        handle_disconnected: bool = False,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Converts the Diffusion Map embedding into an adjacency matrix using the Minimum Spanning Tree (MST) method.

        In this method, an MST is constructed where each node represents a cell, and edges connect
        cells with minimal distances in diffusion space, ensuring that the graph is fully connected with no cycles.

        Mathematical Formulation:
            1.  Construct a complete graph where each node represents a cell.
            2.  The weight of the edge between cell i and cell j is d(i, j) = metric(x_i, x_j).
            3.  Compute the MST of this graph using the specified algorithm (e.g., Kruskal's or Prim's).
            4.  Represent the MST as an adjacency matrix A, where A_{i,j} = d(i, j) if cells i and j are
                connected in the MST; otherwise, A_{i,j} = 0.

        Args:
            metric (str): Distance metric to use. Default is 'euclidean'.
            algorithm (str): The algorithm to compute the MST. Options include 'kruskal'
                and 'prim'. Default is 'kruskal'.
            handle_disconnected (bool): Whether to handle disconnected components by connecting
                them with epsilon-weighted edges. Default is False.
            epsilon (float): The weight to assign to connecting edges when handling
                disconnected components. Default is 1e-5.

        Returns:
            np.ndarray: An adjacency matrix of shape (n_cells, n_cells) representing the MST.

        Raises:
            ValueError: If an unsupported algorithm is specified.
        """
        if algorithm.lower() not in ["kruskal", "prim"]:
            raise ValueError("algorithm must be either 'kruskal' or 'prim'.")

        n_cells = self.diffmap_embedding.shape[0]
        g = nx.Graph()
        g.add_nodes_from(range(n_cells))

        # Compute pairwise distances
        pairwise_dist = squareform(pdist(self.diffmap_embedding, metric=metric))

        # Add edges with weights as distances
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                weight = pairwise_dist[i, j]
                g.add_edge(i, j, weight=weight)

        # Compute the MST
        mst = nx.minimum_spanning_tree(g, algorithm=algorithm.lower())

        # Initialize adjacency matrix
        adjacency = np.zeros((n_cells, n_cells), dtype=float)
        for i, j, data in mst.edges(data=True):
            adjacency[i, j] = data["weight"]
            adjacency[j, i] = data["weight"]  # Ensure symmetry

        if handle_disconnected and not nx.is_connected(mst):
            adjacency = self._handle_disconnected_components(adjacency, epsilon=epsilon)

        self.adjacency_matrix = adjacency
        return self.adjacency_matrix

    def to_gaussian_kernel_adjacency(
        self, sigma: float = 1.0, metric: str = "euclidean", handle_disconnected: bool = True, epsilon: float = 1e-5
    ) -> np.ndarray:
        """Converts the Diffusion Map embedding into an adjacency matrix using a Gaussian Kernel-Based method.

        This method defines edge weights based on a Gaussian kernel applied to the distances
        in diffusion space and then constructs the adjacency matrix.

        Mathematical Formulation:
            1.  Let x_i be the Diffusion Map embedding of cell i.
            2.  Compute the distance d(i, j) = metric(x_i, x_j).
            3.  Define the weight between cells i and j as A_{i,j} = exp(- (d(i, j))^2 / (2 * sigma^2)).
            4.  Set A_{i,j} = 0 if i == j.
            5.  Ensure the adjacency matrix is symmetric.
            6.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            sigma (float): The bandwidth parameter for the Gaussian kernel. Must be positive. Default is 1.0.
            metric (str): Distance metric to use. Default is 'euclidean'.
            handle_disconnected (bool): Whether to handle disconnected components by connecting
                them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when handling
                disconnected components. Default is 1e-5.

        Returns:
            np.ndarray: A weighted adjacency matrix of shape (n_cells, n_cells).

        Raises:
            ValueError: If `sigma` is not a positive number.
        """
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("sigma must be a positive number.")

        # Compute pairwise distances squared
        pairwise_dist_sq = squareform(pdist(self.diffmap_embedding, metric=metric)) ** 2

        # Apply Gaussian kernel
        adjacency = np.exp(-pairwise_dist_sq / (2 * sigma**2))

        # Remove self-connections
        np.fill_diagonal(adjacency, 0)

        if handle_disconnected:
            adjacency = self._handle_disconnected_components(adjacency, epsilon=epsilon)

        self.adjacency_matrix = adjacency
        return self.adjacency_matrix

    def to_adjacency(self, method: str = "knn", **kwargs) -> np.ndarray:
        """Convenience method to compute the adjacency matrix using the specified method.

        Supported Methods:
            - 'knn': k-Nearest Neighbors based adjacency.
            - 'threshold': Threshold-based adjacency.
            - 'mst': Minimum Spanning Tree based adjacency.
            - 'gaussian_kernel': Gaussian Kernel-Based adjacency.

        Args:
            method (str): The adjacency construction method. Default is 'knn'.
            kwargs: Additional keyword arguments for the chosen method.

        Returns:
            np.ndarray: An adjacency matrix of shape (n_cells, n_cells).

        Raises:
            ValueError: If an unsupported method is specified.
        """
        method = method.lower()
        if method == "knn":
            return self.to_knn_adjacency(**kwargs)
        elif method == "threshold":
            return self.to_threshold_adjacency(**kwargs)
        elif method == "mst":
            return self.to_mst_adjacency(**kwargs)
        elif method == "gaussian_kernel":
            return self.to_gaussian_kernel_adjacency(**kwargs)
        else:
            raise ValueError(
                f"Unsupported method {method!r}. Choose from 'knn', 'threshold', 'mst', 'gaussian_kernel'."
            )

    def get_adjacency(self) -> np.ndarray:
        """Retrieves the current adjacency matrix.

        Returns:
            np.ndarray: The current adjacency matrix.

        Raises:
            ValueError: If the adjacency matrix has not been computed yet.
        """
        if self.adjacency_matrix is None:
            raise ValueError(
                "Adjacency matrix has not been computed yet. Call one of the adjacency construction methods first."
            )
        return self.adjacency_matrix
