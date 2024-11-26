#!/usr/bin/env python3

from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


class PseudotimeAdjacencyConverter:
    """A class to convert a pseudotime array into an adjacency matrix using various methods.

    Pseudotime analysis assigns a scalar value to each cell, representing its position along a
    biological trajectory. Converting pseudotime values into an adjacency matrix enables the construction
    of graph-based representations of cellular trajectories, facilitating downstream analyses such as
    trajectory inference, clustering, and network analysis.

    Supported Methods:
        - k-Nearest Neighbors (k-NN) based adjacency
        - Threshold-based adjacency
        - Minimum Spanning Tree (MST) based adjacency
        - Gaussian Kernel-Based Adjacency

    Attributes:
        pseudotime (np.ndarray): Array of pseudotime values for each cell.
        adjacency_matrix (Optional[np.ndarray]): The resulting adjacency matrix after conversion.
    """

    def __init__(self, pseudotime: np.ndarray):
        """Initializes the PseudotimeAdjacencyConverter with a pseudotime array.

        Args:
            pseudotime (np.ndarray): A one-dimensional array of pseudotime values.

        Raises:
            ValueError: If pseudotime is not a one-dimensional numpy array.
            TypeError: If `pseudotime` is not a numpy array.
        """
        if not isinstance(pseudotime, np.ndarray):
            raise TypeError("pseudotime must be a numpy array.")
        if pseudotime.ndim != 1:
            raise ValueError("pseudotime must be a one-dimensional array.")

        self.pseudotime = pseudotime
        self.adjacency_matrix: Optional[np.ndarray] = None

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
        """Converts the pseudotime array into an adjacency matrix using the k-Nearest Neighbors (k-NN) method.

        In this method, each cell is connected to its k nearest neighbors based on the pseudotime values.
        The distance metric can be specified, with 'euclidean' being the default.

        Mathematical Formulation:
            1.  Let t_i be the pseudotime of cell i.
            2.  Compute the distance between cells using the specified metric.
            3.  For each cell i, identify the k cells with the smallest distances to i.
            4.  Set A_{i,j} = distance(i, j) if weighted is True, else A_{i,j} = 1 if j is
                among the k nearest neighbors of i; otherwise, A_{i,j} = 0.
            5.  Ensure the adjacency matrix is symmetric.
            6.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            n_neighbors (int): Number of nearest neighbors to connect. Must be a positive integer. Default is 5.
            metric (str): Distance metric to use. Default is 'euclidean'.
            algorithm (str): Algorithm to compute nearest neighbors. Default is 'auto'.
            include_self (bool): Whether to include the cell itself as its neighbor. Default is False.
            weighted (bool): Whether to assign edge weights based on distance. If
                False, edges are binary. Default is False.
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

        # Reshape pseudotime for NearestNeighbors
        pseudotime_reshaped = self.pseudotime.reshape(-1, 1)

        # Initialize NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + int(include_self), metric=metric, algorithm=algorithm).fit(
            pseudotime_reshaped
        )

        # Find k neighbors for each cell
        distances, indices = nbrs.kneighbors(pseudotime_reshaped)

        n_cells = len(self.pseudotime)
        adjacency = np.zeros((n_cells, n_cells), dtype=float if weighted else int)

        for idx, neighbors in enumerate(indices):
            # Optionally exclude the cell itself from its neighbors
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
        self, delta: float, weighted: bool = False, handle_disconnected: bool = True, epsilon: float = 1e-5
    ) -> np.ndarray:
        """Converts the pseudotime array into an adjacency matrix by thresholding pseudotime differences.

        In this method, two cells are connected if the absolute difference in their pseudotime values
        is less than a specified threshold delta.

        Mathematical Formulation:
            1.  Let t_i and t_j be the pseudotime values of cells i and j, respectively.
            2.  Define A_{i,j} = abs(t_i - t_j) if weighted is True and abs(t_i - t_j) < delta;
                otherwise, A_{i,j} = 1 if abs(t_i - t_j) < delta; else A_{i,j} = 0.
            3.  Ensure the adjacency matrix is symmetric.
            4.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            delta (float): The maximum allowed pseudotime difference for two
                cells to be connected. Must be a positive number.
            weighted (bool): Whether to assign edge weights based on pseudotime
                differences. If False, edges are binary. Default is False.
            handle_disconnected (bool): Whether to handle disconnected components by
                connecting them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when
                handling disconnected components. Default is 1e-5.

        Returns:
            np.ndarray: An adjacency matrix of shape (n_cells, n_cells).

        Raises:
            ValueError: If `delta` is not a positive number.
        """
        if not isinstance(delta, (int, float)) or delta <= 0:
            raise ValueError("delta must be a positive number.")

        n_cells = len(self.pseudotime)
        adjacency = np.zeros((n_cells, n_cells), dtype=float if weighted else int)

        # Compute pairwise absolute differences
        pairwise_diff = squareform(pdist(self.pseudotime.reshape(-1, 1), metric="chebyshev"))

        if weighted:
            adjacency[pairwise_diff < delta] = pairwise_diff[pairwise_diff < delta]
        else:
            adjacency[pairwise_diff < delta] = 1

        # Remove self-connections
        np.fill_diagonal(adjacency, 0)

        if handle_disconnected:
            adjacency = self._handle_disconnected_components(adjacency, epsilon=epsilon)

        self.adjacency_matrix = adjacency
        return self.adjacency_matrix

    def to_mst_adjacency(
        self, algorithm: str = "kruskal", handle_disconnected: bool = False, epsilon: float = 1e-5
    ) -> np.ndarray:
        """Converts the pseudotime array into an adjacency matrix using the Minimum Spanning Tree (MST) method.

        In this method, an MST is constructed where each node represents a cell, and edges connect
        cells with minimal pseudotime differences, ensuring that the graph is fully connected with no cycles.

        Mathematical Formulation:
            1.  Construct a complete graph where each node represents a cell.
            2.  The weight of the edge between cell i and cell j is abs(t_i - t_j).
            3.  Compute the MST of this graph using the specified algorithm (e.g., Kruskal's or Prim's).
            4.  Represent the MST as an adjacency matrix A, where A_{i,j} = abs(t_i - t_j) if
                cells i and j are connected in the MST; otherwise, A_{i,j} = 0.

        Args:
            algorithm (str): The algorithm to compute the MST. Options
                include 'kruskal' and 'prim'. Default is 'kruskal'.
            handle_disconnected (bool): Whether to handle disconnected components by
                connecting them with epsilon-weighted edges. Default is False.
            epsilon (float): The weight to assign to connecting edges when handling
                disconnected components. Default is 1e-5.

        Returns:
            np.ndarray: An adjacency matrix of shape (n_cells, n_cells) representing the MST.

        Raises:
            ValueError: If an unsupported algorithm is specified.
        """
        if algorithm.lower() not in ["kruskal", "prim"]:
            raise ValueError("algorithm must be either 'kruskal' or 'prim'.")

        n_cells = len(self.pseudotime)
        g = nx.Graph()
        g.add_nodes_from(range(n_cells))

        # Add edges with weights as absolute pseudotime differences
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                weight = abs(self.pseudotime[i] - self.pseudotime[j])
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
        self, sigma: float = 1.0, handle_disconnected: bool = True, epsilon: float = 1e-5
    ) -> np.ndarray:
        """Converts the pseudotime array into an adjacency matrix using a Gaussian Kernel-Based method.

        This method defines edge weights based on a Gaussian kernel of the pseudotime differences
        and then constructs the adjacency matrix.

        Mathematical Formulation:
            1.  Let t_i be the pseudotime of cell i.
            2.  Define the weight between cells i and j as A_{i,j} = exp(- (t_i - t_j)^2 / (2 * sigma^2)).
            3.  Set A_{i,j} = 0 if i == j.
            4.  Ensure the adjacency matrix is symmetric.
            5.  Handle disconnected components by thresholding or connecting with epsilon-weighted edges if required.

        Args:
            sigma (float): The bandwidth parameter for the Gaussian kernel. Must be positive. Default is 1.0.
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

        n_cells = len(self.pseudotime)
        adjacency = np.zeros((n_cells, n_cells), dtype=float)

        # Compute pairwise squared differences
        pairwise_diff_sq = squareform(pdist(self.pseudotime.reshape(-1, 1), metric="sqeuclidean"))

        # Apply Gaussian kernel
        adjacency = np.exp(-pairwise_diff_sq / (2 * sigma**2))

        # Remove self-connections
        np.fill_diagonal(adjacency, 0)

        if handle_disconnected:
            adjacency = self._handle_disconnected_components(adjacency, epsilon=epsilon)

        self.adjacency_matrix = adjacency
        return self.adjacency_matrix

    def _validate_adjacency_matrix_shape(self, n_labels: int):
        """Validates the shape of the adjacency matrix.

        Args:
            n_labels (int): Number of unique labels.

        Raises:
            ValueError: If the adjacency matrix shape does not match (n_labels, n_labels).
        """
        if self.adjacency_matrix is not None:
            if self.adjacency_matrix.shape != (n_labels, n_labels):
                raise ValueError(
                    f"Adjacency matrix shape {self.adjacency_matrix.shape} does not match the number of labels ({n_labels})."
                )

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


class PseudotimeLabelAdjacencyConverter:
    """A class to convert a cell-level pseudotime array into a label-level adjacency matrix.

    This involves aggregating pseudotime values per label and then constructing an adjacency matrix
    based on the relationships between labels derived from cell pseudotime.

    Supported Methods for Aggregation:
        - 'mean': Average pseudotime per label.
        - 'median': Median pseudotime per label.
        - 'min': Minimum pseudotime per label.
        - 'max': Maximum pseudotime per label.
        - 'trimmed_mean': Trimmed mean pseudotime per label.

    Supported Methods for Adjacency Construction:
        - 'knn': k-Nearest Neighbors based adjacency.
        - 'threshold': Threshold-based adjacency.
        - 'mst': Minimum Spanning Tree based adjacency.
        - 'gaussian_kernel': Gaussian Kernel-Based adjacency (New Method).

    Attributes:
        cell_pseudotime (np.ndarray): Array of pseudotime values for each cell (n,).
        cell_labels (np.ndarray): Array mapping each cell to its corresponding label (n,).
        label_adjacency_matrix (Optional[np.ndarray]): The resulting label-level adjacency matrix (l x l).
        labels (np.ndarray): Array of unique labels.
        aggregated_pseudotime (Optional[np.ndarray]): Aggregated pseudotime values per label.
    """

    def __init__(self, cell_pseudotime: np.ndarray, cell_labels: np.ndarray, label_adjacency_matrix_labels: np.ndarray):
        """Initializes the converter with a cell pseudotime array and cell labels.

        Args:
            cell_pseudotime (np.ndarray): A one-dimensional array of pseudotime values for each cell (n,).
            cell_labels (np.ndarray): A one-dimensional array of labels for each cell (n,).
            label_adjacency_matrix_labels (np.ndarray): A one-dimensional array of labels for each label (l,).

        Raises:
            ValueError: If dimensions of the inputs are invalid or if labels are inconsistent.
            TypeError: If `cell_labels` or `cell_pseudotime` are not numpy arrays.
        """
        if not isinstance(cell_pseudotime, np.ndarray):
            raise TypeError("cell_pseudotime must be a numpy array.")
        if cell_pseudotime.ndim != 1:
            raise ValueError("cell_pseudotime must be a one-dimensional array.")
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a numpy array.")
        if cell_labels.ndim != 1:
            raise ValueError("cell_labels must be a one-dimensional array.")
        if len(cell_pseudotime) != len(cell_labels):
            raise ValueError("cell_pseudotime and cell_labels must have the same length.")

        self.cell_pseudotime = cell_pseudotime
        self.cell_labels = cell_labels
        self.label_adjacency_matrix: Optional[np.ndarray] = None
        self.adj_labels = label_adjacency_matrix_labels
        self.aggregated_pseudotime: Optional[np.ndarray] = None

    def aggregate_pseudotime_per_label(self, aggregation: str = "mean", trim_percent: float = 0.1) -> np.ndarray:
        """Aggregates pseudotime values per label using the specified method.

        Supported Aggregation Methods:
            - 'mean': Average pseudotime per label.
            - 'median': Median pseudotime per label.
            - 'min': Minimum pseudotime per label.
            - 'max': Maximum pseudotime per label.
            - 'trimmed_mean': Trimmed mean pseudotime per label.

        Args:
            aggregation (str): The aggregation method to use. Default is 'mean'.
            trim_percent (float): The proportion of data to trim from each end for
                'trimmed_mean'. Must be between 0 and 0.5. Default is 0.1.

        Returns:
            np.ndarray: A one-dimensional array of aggregated pseudotime values per label (l,).

        Raises:
            ValueError: If an unsupported aggregation method is specified or if `trim_percent` is invalid.
        """
        aggregation = aggregation.lower()
        aggregated_pseudotime = np.zeros(len(self.adj_labels))

        for idx, label in enumerate(self.adj_labels):
            label_pseudotime = self.cell_pseudotime[self.cell_labels == label]
            if aggregation == "mean":
                aggregated_pseudotime[idx] = label_pseudotime.mean()
            elif aggregation == "median":
                aggregated_pseudotime[idx] = np.median(label_pseudotime)
            elif aggregation == "min":
                aggregated_pseudotime[idx] = label_pseudotime.min()
            elif aggregation == "max":
                aggregated_pseudotime[idx] = label_pseudotime.max()
            elif aggregation == "trimmed_mean":
                if not (0 < trim_percent < 0.5):
                    raise ValueError("trim_percent must be between 0 and 0.5 for 'trimmed_mean' aggregation.")
                sorted_pseudotime = np.sort(label_pseudotime)
                trim_size = int(len(sorted_pseudotime) * trim_percent)
                trimmed = sorted_pseudotime[trim_size:-trim_size] if trim_size > 0 else sorted_pseudotime
                aggregated_pseudotime[idx] = trimmed.mean()
            else:
                raise ValueError(
                    f"Unsupported aggregation method {aggregation!r}. Choose from 'mean', 'median', 'min', 'max', 'trimmed_mean'."
                )

        self.aggregated_pseudotime = aggregated_pseudotime
        return self.aggregated_pseudotime

    def construct_label_adjacency(self, method: str = "knn", **kwargs) -> np.ndarray:
        """Constructs a label-level adjacency matrix based on aggregated pseudotime values.

        Supported Methods:
            - 'knn': k-Nearest Neighbors based adjacency.
            - 'threshold': Threshold-based adjacency.
            - 'mst': Minimum Spanning Tree based adjacency.
            - 'gaussian_kernel': Gaussian Kernel-Based adjacency.

        Args:
            method (str): The method to construct adjacency. Default is 'knn'.
            kwargs: Additional keyword arguments for the chosen method.

        Returns:
            np.ndarray: A binary adjacency matrix of shape (l x l).

        Raises:
            ValueError: If an unsupported method is specified or if aggregated pseudotime has not been computed.
        """
        if self.aggregated_pseudotime is None:
            raise ValueError(
                "Aggregated pseudotime per label has not been computed. Call aggregate_pseudotime_per_label() first."
            )

        method = method.lower()
        converter = PseudotimeAdjacencyConverter(pseudotime=self.aggregated_pseudotime)

        if method == "knn":
            n_neighbors = kwargs.get("n_neighbors", 2)
            metric = kwargs.get("metric", "euclidean")
            algorithm = kwargs.get("algorithm", "auto")
            label_adjacency = converter.to_knn_adjacency(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
        elif method == "threshold":
            delta = kwargs.get("delta", 1.0)
            label_adjacency = converter.to_threshold_adjacency(delta=delta)
        elif method == "mst":
            label_adjacency = converter.to_mst_adjacency()
        elif method == "gaussian_kernel":
            sigma = kwargs.get("sigma", 1.0)
            label_adjacency = converter.to_gaussian_kernel_adjacency(sigma=sigma)
        else:
            raise ValueError(
                f"Unsupported method {method!r}. Choose from 'knn', 'threshold', 'mst', 'gaussian_kernel'."
            )

        self.label_adjacency_matrix = label_adjacency
        return self.label_adjacency_matrix

    def get_label_adjacency(
        self,
        aggregation: str = "mean",
        aggregation_kwargs: Optional[dict] = None,
        method: str = "knn",
        adjacency_kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """Convenience method to aggregate pseudotime and construct label adjacency.

        Args:
            aggregation (str): The aggregation method to use. Default is 'mean'.
            aggregation_kwargs (Optional[dict]): Additional keyword arguments for aggregation methods.
            method (str): The method to construct adjacency. Default is 'knn'.
            adjacency_kwargs (Optional[dict]): Additional keyword arguments for adjacency construction methods.

        Returns:
            np.ndarray: A binary adjacency matrix of shape (l x l).
        """
        if aggregation_kwargs is None:
            aggregation_kwargs = {}
        if adjacency_kwargs is None:
            adjacency_kwargs = {}

        self.aggregate_pseudotime_per_label(aggregation=aggregation, **aggregation_kwargs)
        return self.construct_label_adjacency(method=method, **adjacency_kwargs)
