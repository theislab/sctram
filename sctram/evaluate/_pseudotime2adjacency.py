#!/usr/bin/env python3

from typing import Optional

import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csgraph
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
        """
        Handles disconnected components in the adjacency matrix by connecting them with epsilon-weighted edges.

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
            4.  Set A_{i,j} = distance(i, j) if weighted is True, else A_{i,j} = 1 if j is among the k nearest neighbors of i; otherwise, A_{i,j} = 0.
            5.  Ensure the adjacency matrix is symmetric.
            6.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            n_neighbors (int): Number of nearest neighbors to connect. Must be a positive integer. Default is 5.
            metric (str): Distance metric to use. Default is 'euclidean'.
            algorithm (str): Algorithm to compute nearest neighbors. Default is 'auto'.
            include_self (bool): Whether to include the cell itself as its neighbor. Default is False.
            weighted (bool): Whether to assign edge weights based on distance. If False, edges are binary. Default is False.
            handle_disconnected (bool): Whether to handle disconnected components by connecting them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when handling disconnected components. Default is 1e-5.

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
            2.  Define A_{i,j} = |t_i - t_j| if weighted is True and |t_i - t_j| < delta; otherwise, A_{i,j} = 1 if |t_i - t_j| < delta; else A_{i,j} = 0.
            3.  Ensure the adjacency matrix is symmetric.
            4.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            delta (float): The maximum allowed pseudotime difference for two cells to be connected. Must be a positive number.
            weighted (bool): Whether to assign edge weights based on pseudotime differences. If False, edges are binary. Default is False.
            handle_disconnected (bool): Whether to handle disconnected components by connecting them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when handling disconnected components. Default is 1e-5.

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
            2.  The weight of the edge between cell i and cell j is |t_i - t_j|.
            3.  Compute the MST of this graph using the specified algorithm (e.g., Kruskal's or Prim's).
            4.  Represent the MST as an adjacency matrix A, where A_{i,j} = |t_i - t_j| if cells i and j are connected in the MST; otherwise, A_{i,j} = 0.

        Args:
            algorithm (str): The algorithm to compute the MST. Options include 'kruskal' and 'prim'. Default is 'kruskal'.
            handle_disconnected (bool): Whether to handle disconnected components by connecting them with epsilon-weighted edges. Default is False.
            epsilon (float): The weight to assign to connecting edges when handling disconnected components. Default is 1e-5.

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
            handle_disconnected (bool): Whether to handle disconnected components by connecting them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when handling disconnected components. Default is 1e-5.

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
                f"Unsupported method '{method}'. Choose from 'knn', 'threshold', 'mst', 'gaussian_kernel'."
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


class AdjacencyPseudotimeConverter:
    """A class to convert an adjacency matrix into a pseudotime array using various methods.

    Trajectory inference aims to order cells along a developmental or differentiation pathway,
    assigning a pseudotime value to each cell that reflects its progress along the trajectory.
    Converting an adjacency matrix into pseudotime values enables the quantification of
    cellular progression based on graph-based relationships.

    Supported Methods:
        - Shortest Path-Based Pseudotime
        - Diffusion-Based Pseudotime
        - Spectral Ordering Pseudotime
        - Gaussian Kernel-Based Pseudotime (New Method)

    Attributes:
        adjacency_matrix (np.ndarray): The input adjacency matrix representing cell connections.
        graph (networkx.Graph): The graph constructed from the adjacency matrix.
        pseudotime (Optional[np.ndarray]): The resulting pseudotime array after conversion.
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        """Initializes the AdjacencyPseudotimeConverter with an adjacency matrix.

        Args:
            adjacency_matrix (np.ndarray): A two-dimensional binary or weighted adjacency matrix.

        Raises:
            ValueError: If `adjacency_matrix` is not a two-dimensional square numpy array.
            TypeError: If `adjacency_matrix` is not a numpy array.
        """
        if not isinstance(adjacency_matrix, np.ndarray):
            raise TypeError("adjacency_matrix must be a numpy array.")
        if adjacency_matrix.ndim != 2:
            raise ValueError("adjacency_matrix must be a two-dimensional array.")
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be a square matrix.")

        self.adjacency_matrix = adjacency_matrix
        self.graph = nx.from_numpy_array(adjacency_matrix)

    def _ensure_distance_weights(self):
        """
        Ensures that the adjacency matrix represents distances.
        If the adjacency matrix represents similarities, it transforms them into distances.
        This is a private method used internally to prepare the graph for algorithms that require distance metrics.
        """
        # Check if any weights are greater than 1, assuming similarities are <=1 and distances >=0
        if np.any(self.adjacency_matrix > 1):
            # Assuming weights are similarities, convert to distances
            max_weight = self.adjacency_matrix.max()
            self.adjacency_matrix = max_weight - self.adjacency_matrix
            self.graph = nx.from_numpy_array(self.adjacency_matrix)
        else:
            # If weights are already distances, ensure non-negativity
            if np.any(self.adjacency_matrix < 0):
                raise ValueError("Adjacency matrix contains negative weights, which are invalid for distance metrics.")

    def to_shortest_path_pseudotime(
        self,
        root_cell: int = 0,
        weight: Optional[str] = "weight",
        handle_disconnected: str = "assign_max_plus_one",
        alternative_distance: Optional[float] = None,
    ) -> np.ndarray:
        """Converts the adjacency matrix into a pseudotime array using the Shortest Path-Based method.

        In this method, pseudotime values are assigned based on the shortest path lengths
        from a specified root cell to all other cells in the graph.

        Mathematical Formulation:
            1.  Let G = (V, E) be the graph where V is the set of cells and E is the set of edges.
            2.  Choose a root cell r ∈ V.
            3.  Compute the shortest path length d(r, v) for each v ∈ V using Dijkstra's algorithm.
            4.  Assign pseudotime(v) = d(r, v).

        Args:
            root_cell (int): The index of the root cell from which pseudotime is calculated. Default is 0.
            weight (Optional[str]): Edge attribute to use as weight.
                If None, unweighted shortest paths are computed. Default is 'weight'.
            handle_disconnected (str): Strategy to handle disconnected cells. Options:
                - 'assign_max_plus_one': Assign max finite pseudotime plus one to disconnected cells.
                - 'assign_alternative_distance': Assign a specified alternative distance to disconnected cells.
                - 'ignore': Leave disconnected cells with pseudotime as infinity.
            alternative_distance (Optional[float]): The distance value to assign to disconnected cells if
                `handle_disconnected` is set to 'assign_alternative_distance'. Must be provided in this case.

        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.

        Raises:
            ValueError: If `root_cell` is not a valid node index, or if `handle_disconnected` is invalid.
        """
        if root_cell < 0 or root_cell >= self.adjacency_matrix.shape[0]:
            raise ValueError(f"root_cell must be between 0 and {self.adjacency_matrix.shape[0] - 1}.")

        # Ensure weights represent distances
        if weight is not None:
            self._ensure_distance_weights()

        try:
            lengths = nx.single_source_dijkstra_path_length(self.graph, root_cell, weight=weight)
        except nx.NetworkXNoPath as err:
            raise ValueError(f"No path found from root_cell {root_cell} to other cells.") from err

        # Initialize pseudotime with infinities
        pseudotime = np.full(self.adjacency_matrix.shape[0], np.inf)

        # Assign computed lengths
        for cell, length in lengths.items():
            pseudotime[cell] = length

        # Handle disconnected cells
        if handle_disconnected == "assign_max_plus_one":
            max_finite = np.max(pseudotime[np.isfinite(pseudotime)])
            pseudotime[np.isinf(pseudotime)] = max_finite + 1
        elif handle_disconnected == "assign_alternative_distance":
            if alternative_distance is None:
                raise ValueError(
                    "alternative_distance must be provided when handle_disconnected is 'assign_alternative_distance'."
                )
            pseudotime[np.isinf(pseudotime)] = alternative_distance
        elif handle_disconnected == "ignore":
            pass  # Leave infinities as is
        else:
            raise ValueError(
                "handle_disconnected must be one of 'assign_max_plus_one', 'assign_alternative_distance', or 'ignore'."
            )

        return pseudotime

    def to_diffusion_pseudotime(
        self, alpha: float = 0.5, n_steps: int = 100, tol: float = 1e-6, root_cell: int = 0
    ) -> np.ndarray:
        """Converts the adjacency matrix into a pseudotime array using the Diffusion-Based method.

        This method models the diffusion process on the graph and assigns pseudotime values based on
        the steady-state distribution of the diffusion process starting from the root cell.

        Mathematical Formulation:
            1.  Let G = (V, E) be the graph with adjacency matrix A.
            2.  Compute the transition probability matrix P where P = D^{-1} A, and D is the degree matrix.
            3.  Simulate the diffusion process with damping factor alpha:
                -   F = alpha * P^T F + (1 - alpha) e_r
                -   where e_r is the initial distribution (one-hot vector for root_cell).
            4.  Iterate until convergence to obtain the steady-state distribution F.
            5.  Assign pseudotime(v) = F[v].

        Args:
            alpha (float): Damping factor controlling the influence of the diffusion.
                Must be between 0 and 1 (exclusive). Default is 0.5.
            n_steps (int): Maximum number of diffusion steps. Default is 100.
            tol (float): Tolerance for convergence. Iterations stop when the change is below this value. Default is 1e-6.
            root_cell (int): The index of the root cell from which diffusion starts. Default is 0.

        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.

        Raises:
            ValueError: If `alpha` is not between 0 and 1, if `root_cell` is invalid, or if the adjacency matrix contains isolated nodes.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be strictly between 0 and 1.")
        if root_cell < 0 or root_cell >= self.adjacency_matrix.shape[0]:
            raise ValueError(f"root_cell must be between 0 and {self.adjacency_matrix.shape[0] - 1}.")

        # Compute the degree matrix
        degrees = self.adjacency_matrix.sum(axis=1)
        if np.any(degrees == 0):
            raise ValueError("The adjacency matrix contains isolated nodes with zero degree.")

        # Compute the transition probability matrix P
        P = self.adjacency_matrix / degrees[:, np.newaxis]

        # Initialize the steady-state distribution F
        F = np.zeros(self.adjacency_matrix.shape[0])
        F[root_cell] = 1.0  # Start diffusion from the root cell

        # Initialize e_r
        e_r = np.zeros(self.adjacency_matrix.shape[0])
        e_r[root_cell] = 1.0

        for step in range(n_steps):
            F_new = alpha * P.T.dot(F) + (1 - alpha) * e_r
            # Check for convergence
            if np.linalg.norm(F_new - F, ord=1) < tol:
                F = F_new
                print(f"Diffusion pseudotime converged in {step + 1} steps.")
                break
            F = F_new
        else:
            print(f"Diffusion pseudotime did not converge within {n_steps} steps.")

        return F

    def to_spectral_pseudotime(
        self, n_components: int = 2, root_cell: Optional[int] = None, normalized: bool = False
    ) -> np.ndarray:
        """Converts the adjacency matrix into a pseudotime array using the Spectral Ordering method.

        This method utilizes spectral embedding by computing the eigenvectors of the graph Laplacian
        and assigns pseudotime based on the projection of cells onto the principal eigenvector.

        Mathematical Formulation:
            1.  Let G = (V, E) be the graph with adjacency matrix A.
            2.  Compute the graph Laplacian L. If normalized is True, use the normalized Laplacian.
            3.  Compute the first `n_components` non-trivial eigenvectors of L.
            4.  Use the Fiedler vector (second smallest eigenvector) to assign pseudotime.
            5.  Normalize the Fiedler vector to range [0, 1].
            6.  Optionally, orient the pseudotime based on the root cell.

        Args:
            n_components (int): Number of eigenvectors to compute. Default is 2.
            root_cell (Optional[int]): The index of the root cell to orient the pseudotime. If None,
                pseudotime is assigned based on the Fiedler vector. Default is None.
            normalized (bool): Whether to use the normalized graph Laplacian. Default is False.

        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.

        Raises:
            ValueError: If `n_components` is not a positive integer, if `root_cell` is invalid,
                        or if the graph is disconnected and `n_components` < number of connected components.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        if root_cell is not None:
            if root_cell < 0 or root_cell >= self.adjacency_matrix.shape[0]:
                raise ValueError(f"root_cell must be between 0 and {self.adjacency_matrix.shape[0] - 1}.")

        # Compute the graph Laplacian
        laplacian = csgraph.laplacian(self.adjacency_matrix, normed=normalized)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(laplacian)

        # Check for connectedness
        num_connected_components = np.sum(np.isclose(eigenvalues, 0))
        if n_components < num_connected_components:
            raise ValueError(
                f"n_components={n_components} is less than the number of connected components ({num_connected_components})."
            )

        if n_components >= 2:
            # Select the Fiedler vector (second smallest eigenvector)
            fiedler_vector = eigenvectors[:, 1]
        else:
            # If n_components=1, use the first non-trivial eigenvector
            fiedler_vector = eigenvectors[:, 0]

        # Normalize the Fiedler vector to range [0, 1]
        min_val = fiedler_vector.min()
        max_val = fiedler_vector.max()
        if max_val - min_val == 0:
            raise ValueError("Fiedler vector has zero variance; cannot normalize.")
        fiedler_norm = (fiedler_vector - min_val) / (max_val - min_val)

        pseudotime = fiedler_norm

        if root_cell is not None:
            # Orient the pseudotime based on the root cell
            direction = pseudotime[root_cell]
            if direction < 0.5:
                pseudotime = 1 - pseudotime  # Flip the direction

        return pseudotime

    def to_gaussian_kernel_pseudotime(self, sigma: float = 1.0, root_cell: int = 0) -> np.ndarray:
        """Converts the adjacency matrix into a pseudotime array using a Gaussian Kernel-Based method.

        This method defines edge weights based on a Gaussian kernel of the pseudotime differences
        and then performs spectral ordering based on the resulting weighted graph.

        Mathematical Formulation:
            1.  Let t_i be the pseudotime of cell i.
            2.  Define the weight between cells i and j as A_{i,j} = exp(- (t_i - t_j)^2 / (2 * sigma^2)).
            3.  Construct the adjacency matrix using these weights.
            4.  Perform spectral ordering on the weighted adjacency matrix to assign pseudotime.

        Args:
            sigma (float): The bandwidth parameter for the Gaussian kernel. Must be positive. Default is 1.0.
            root_cell (int): The index of the root cell to orient the pseudotime. Default is 0.

        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.

        Raises:
            ValueError: If `sigma` is not positive or if `root_cell` is invalid.
        """
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("sigma must be a positive number.")
        if root_cell < 0 or root_cell >= self.adjacency_matrix.shape[0]:
            raise ValueError(f"root_cell must be between 0 and {self.adjacency_matrix.shape[0] - 1}.")

        # Compute pseudotime using shortest path as a preliminary step
        preliminary_pseudotime = self.to_shortest_path_pseudotime(root_cell=root_cell)

        # Compute Gaussian kernel weights
        diff_matrix = np.abs(preliminary_pseudotime[:, np.newaxis] - preliminary_pseudotime[np.newaxis, :])
        gaussian_weights = np.exp(-(diff_matrix**2) / (2 * sigma**2))

        # Update the graph with Gaussian weights
        gaussian_weights_matrix = gaussian_weights * self.adjacency_matrix
        gaussian_graph = nx.from_numpy_array(gaussian_weights_matrix)

        # Perform spectral ordering on the Gaussian-weighted graph
        laplacian = csgraph.laplacian(gaussian_weights_matrix, normed=True)
        eigenvalues, eigenvectors = eigh(laplacian)

        # Select the Fiedler vector
        if len(eigenvalues) < 2:
            raise ValueError("Graph must have at least two eigenvalues for spectral pseudotime.")
        fiedler_vector = eigenvectors[:, 1]

        # Normalize the Fiedler vector to range [0, 1]
        min_val = fiedler_vector.min()
        max_val = fiedler_vector.max()
        if max_val - min_val == 0:
            raise ValueError("Fiedler vector has zero variance; cannot normalize.")
        fiedler_norm = (fiedler_vector - min_val) / (max_val - min_val)

        pseudotime = fiedler_norm

        # Orient based on root cell
        direction = pseudotime[root_cell]
        if direction < 0.5:
            pseudotime = 1 - pseudotime

        return pseudotime

    def is_connected(self) -> bool:
        """Checks if the graph is connected.

        Returns:
            bool: True if the graph is connected, False otherwise.
        """
        return nx.is_connected(self.graph)

    def get_connected_components(self) -> list:
        """Retrieves the connected components of the graph.

        Returns:
            list: A list of sets, each containing the nodes in a connected component.
        """
        return list(nx.connected_components(self.graph))

    def to_pseudotime(self, method: str = "shortest_path", **kwargs) -> np.ndarray:
        """Convenience method to compute pseudotime using the specified method.

        Supported Methods:
            - 'shortest_path': Shortest Path-Based Pseudotime.
            - 'diffusion': Diffusion-Based Pseudotime.
            - 'spectral': Spectral Ordering Pseudotime.
            - 'gaussian_kernel': Gaussian Kernel-Based Pseudotime.

        Args:
            method (str): The pseudotime computation method. Default is 'shortest_path'.
            kwargs: Additional keyword arguments for the chosen method.

        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.

        Raises:
            ValueError: If an unsupported method is specified.
        """
        method = method.lower()
        if method == "shortest_path":
            return self.to_shortest_path_pseudotime(**kwargs)
        elif method == "diffusion":
            return self.to_diffusion_pseudotime(**kwargs)
        elif method == "spectral":
            return self.to_spectral_pseudotime(**kwargs)
        elif method == "gaussian_kernel":
            return self.to_gaussian_kernel_pseudotime(**kwargs)
        else:
            raise ValueError(
                f"Unsupported method '{method}'. Choose from 'shortest_path', 'diffusion', 'spectral', 'gaussian_kernel'."
            )


class LabelAdjacencyPseudotimeConverter:
    """A class to convert a label-level adjacency matrix into a cell-level pseudotime array.

    This involves two main steps:
        1. Compute pseudotime for each label using graph-based methods.
        2. Assign pseudotime to each cell based on its label's pseudotime.

    Supported Methods for Pseudotime Computation:
        - Shortest Path-Based Pseudotime
        - Diffusion-Based Pseudotime
        - Spectral Ordering Pseudotime
        - Gaussian Kernel-Based Pseudotime

    Attributes:
        label_adjacency_matrix (np.ndarray): The input label-level adjacency matrix (l x l).
        cell_labels (np.ndarray): Array mapping each cell to its corresponding label.
        label_pseudotime (Optional[np.ndarray]): The resulting pseudotime array for labels.
        cell_pseudotime (Optional[np.ndarray]): The resulting pseudotime array for cells.
    """

    def __init__(self, label_adjacency_matrix: np.ndarray, cell_labels: np.ndarray):
        """Initializes the converter with a label adjacency matrix and cell labels.

        Args:
            label_adjacency_matrix (np.ndarray): A two-dimensional binary or weighted adjacency matrix (l x l).
            cell_labels (np.ndarray): A one-dimensional array of labels for each cell (n,).

        Raises:
            ValueError: If dimensions of the adjacency matrix or cell_labels are invalid.
            TypeError: If `label_adjacency_matrix` or `cell_labels` are not numpy arrays.
        """
        if not isinstance(label_adjacency_matrix, np.ndarray):
            raise TypeError("label_adjacency_matrix must be a numpy array.")
        if label_adjacency_matrix.ndim != 2 or label_adjacency_matrix.shape[0] != label_adjacency_matrix.shape[1]:
            raise ValueError("label_adjacency_matrix must be a square two-dimensional array.")
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a numpy array.")
        if cell_labels.ndim != 1:
            raise ValueError("cell_labels must be a one-dimensional array.")
        unique_labels = np.unique(cell_labels)
        if label_adjacency_matrix.shape[0] != len(unique_labels):
            raise ValueError("Unique cell_labels do not match the dimensions of adjacency matrix.")

        self.label_adjacency_matrix = label_adjacency_matrix
        self.cell_labels = cell_labels
        self.label_pseudotime: Optional[np.ndarray] = None
        self.cell_pseudotime: Optional[np.ndarray] = None
        self.labels = unique_labels

    def get_label_pseudotime(
        self,
        method: str = "shortest_path",
        handle_disconnected: str = "assign_max_plus_one",
        alternative_distance: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Computes pseudotime for each label using the specified method.

        Supported Methods:
            - 'shortest_path': Shortest Path-Based Pseudotime.
            - 'diffusion': Diffusion-Based Pseudotime.
            - 'spectral': Spectral Ordering Pseudotime.
            - 'gaussian_kernel': Gaussian Kernel-Based Pseudotime.

        Args:
            method (str): The pseudotime computation method. Default is 'shortest_path'.
            handle_disconnected (str): Strategy to handle disconnected labels. Options:
                - 'assign_max_plus_one': Assign max finite pseudotime plus one to disconnected labels.
                - 'assign_alternative_distance': Assign a specified alternative distance to disconnected labels.
                - 'ignore': Leave disconnected labels with pseudotime as infinity.
            alternative_distance (Optional[float]): The distance value to assign to disconnected labels if
                `handle_disconnected` is set to 'assign_alternative_distance'. Must be provided in this case.
            kwargs: Additional keyword arguments for the chosen method.

        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each label.

        Raises:
            ValueError: If an unsupported method is specified or required parameters are missing.
        """
        method = method.lower()
        converter = AdjacencyPseudotimeConverter(self.label_adjacency_matrix)

        if method == "shortest_path":
            root_label = kwargs.get("root_label", 0)
            self.label_pseudotime = converter.to_shortest_path_pseudotime(
                root_cell=root_label,
                weight="weight",
                handle_disconnected=handle_disconnected,
                alternative_distance=alternative_distance,
            )
        elif method == "diffusion":
            alpha = kwargs.get("alpha", 0.5)
            n_steps = kwargs.get("n_steps", 100)
            tol = kwargs.get("tol", 1e-6)
            root_label = kwargs.get("root_label", 0)
            self.label_pseudotime = converter.to_diffusion_pseudotime(
                alpha=alpha, n_steps=n_steps, tol=tol, root_cell=root_label
            )
        elif method == "spectral":
            n_components = kwargs.get("n_components", 2)
            root_label = kwargs.get("root_label", None)
            normalized = kwargs.get("normalized", False)
            self.label_pseudotime = converter.to_spectral_pseudotime(
                n_components=n_components, root_cell=root_label, normalized=normalized
            )
        elif method == "gaussian_kernel":
            sigma = kwargs.get("sigma", 1.0)
            root_label = kwargs.get("root_label", 0)
            self.label_pseudotime = converter.to_gaussian_kernel_pseudotime(sigma=sigma, root_cell=root_label)
        else:
            raise ValueError(
                f"Unsupported method '{method}'. Choose from 'shortest_path', 'diffusion', 'spectral', 'gaussian_kernel'."
            )

        return self.label_pseudotime

    def assign_cell_pseudotime(
        self, handle_disconnected: str = "assign_max_plus_one", alternative_distance: Optional[float] = None
    ) -> np.ndarray:
        """Assigns pseudotime to each cell based on its label's pseudotime.

        Args:
            handle_disconnected (str): Strategy to handle disconnected labels. Must match the strategy used in `get_label_pseudotime`.
            alternative_distance (Optional[float]): The distance value to assign to disconnected labels if
                `handle_disconnected` is set to 'assign_alternative_distance'. Must be provided in this case.

        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.

        Raises:
            ValueError: If label pseudotime has not been computed or if `handle_disconnected` is invalid.
        """
        if self.label_pseudotime is None:
            raise ValueError("Label pseudotime has not been computed. Call `get_label_pseudotime` first.")

        # Map each label to its pseudotime
        label_to_pseudotime = {label: self.label_pseudotime[idx] for idx, label in enumerate(self.labels)}

        # Assign pseudotime to each cell based on its label
        self.cell_pseudotime = np.vectorize(label_to_pseudotime.get)(self.cell_labels)

        # Handle cells with labels that might not have been assigned pseudotime (if any)
        if handle_disconnected == "assign_max_plus_one":
            max_pseudotime = np.max(self.label_pseudotime)
            self.cell_pseudotime = np.where(np.isinf(self.cell_pseudotime), max_pseudotime + 1, self.cell_pseudotime)
        elif handle_disconnected == "assign_alternative_distance":
            if alternative_distance is None:
                raise ValueError(
                    "alternative_distance must be provided when handle_disconnected is 'assign_alternative_distance'."
                )
            self.cell_pseudotime = np.where(np.isinf(self.cell_pseudotime), alternative_distance, self.cell_pseudotime)
        elif handle_disconnected == "ignore":
            pass  # Leave infinities as is
        else:
            raise ValueError(
                "handle_disconnected must be one of 'assign_max_plus_one', 'assign_alternative_distance', or 'ignore'."
            )

        return self.cell_pseudotime

    def get_cell_pseudotime(
        self,
        method: str = "shortest_path",
        handle_disconnected: str = "assign_max_plus_one",
        alternative_distance: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Convenience method to compute label pseudotime and assign it to cells.

        Args:
            method (str): The pseudotime computation method. Default is 'shortest_path'.
            handle_disconnected (str): Strategy to handle disconnected labels. Default is 'assign_max_plus_one'.
            alternative_distance (Optional[float]): The distance value to assign to disconnected labels if
                `handle_disconnected` is set to 'assign_alternative_distance'. Must be provided in this case.
            kwargs: Additional keyword arguments for the pseudotime computation method.

        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.
        """
        self.get_label_pseudotime(
            method=method, handle_disconnected=handle_disconnected, alternative_distance=alternative_distance, **kwargs
        )
        return self.assign_cell_pseudotime(
            handle_disconnected=handle_disconnected, alternative_distance=alternative_distance
        )


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

    def __init__(self, cell_pseudotime: np.ndarray, cell_labels: np.ndarray):
        """Initializes the converter with a cell pseudotime array and cell labels.

        Args:
            cell_pseudotime (np.ndarray): A one-dimensional array of pseudotime values for each cell (n,).
            cell_labels (np.ndarray): A one-dimensional array of labels for each cell (n,).

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
        self.labels = np.unique(cell_labels)
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
            trim_percent (float): The proportion of data to trim from each end for 'trimmed_mean'. Must be between 0 and 0.5. Default is 0.1.

        Returns:
            np.ndarray: A one-dimensional array of aggregated pseudotime values per label (l,).

        Raises:
            ValueError: If an unsupported aggregation method is specified or if `trim_percent` is invalid.
        """
        aggregation = aggregation.lower()
        aggregated_pseudotime = np.zeros(len(self.labels))

        for idx, label in enumerate(self.labels):
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
                    f"Unsupported aggregation method '{aggregation}'. Choose from 'mean', 'median', 'min', 'max', 'trimmed_mean'."
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
                f"Unsupported method '{method}'. Choose from 'knn', 'threshold', 'mst', 'gaussian_kernel'."
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

        Raises:
            ValueError: If invalid parameters are provided.
        """
        if aggregation_kwargs is None:
            aggregation_kwargs = {}
        if adjacency_kwargs is None:
            adjacency_kwargs = {}

        self.aggregate_pseudotime_per_label(aggregation=aggregation, **aggregation_kwargs)
        return self.construct_label_adjacency(method=method, **adjacency_kwargs)


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
        """
        Handles disconnected components in the adjacency matrix by connecting them with epsilon-weighted edges.

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
            4.  Set A_{i,j} = distance(i, j) if weighted is True, else A_{i,j} = 1 if j is among the k nearest neighbors of i; otherwise, A_{i,j} = 0.
            5.  Ensure the adjacency matrix is symmetric.
            6.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            n_neighbors (int): Number of nearest neighbors to connect. Must be a positive integer. Default is 5.
            metric (str): Distance metric to use. Default is 'euclidean'.
            algorithm (str): Algorithm to compute nearest neighbors. Default is 'auto'.
            include_self (bool): Whether to include the cell itself as its neighbor. Default is False.
            weighted (bool): Whether to assign edge weights based on distance. If False, edges are binary. Default is False.
            handle_disconnected (bool): Whether to handle disconnected components by connecting them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when handling disconnected components. Default is 1e-5.

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
            3.  Define A_{i,j} = d(i, j) if weighted is True and d(i, j) < delta; otherwise, A_{i,j} = 1 if d(i, j) < delta; else A_{i,j} = 0.
            4.  Ensure the adjacency matrix is symmetric.
            5.  Handle disconnected components by connecting them with epsilon-weighted edges if required.

        Args:
            delta (float): The maximum allowed distance for two cells to be connected. Must be a positive number.
            metric (str): Distance metric to use. Default is 'euclidean'.
            weighted (bool): Whether to assign edge weights based on distance. If False, edges are binary. Default is False.
            handle_disconnected (bool): Whether to handle disconnected components by connecting them with epsilon-weighted edges. Default is True.
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
            4.  Represent the MST as an adjacency matrix A, where A_{i,j} = d(i, j) if cells i and j are connected in the MST; otherwise, A_{i,j} = 0.

        Args:
            metric (str): Distance metric to use. Default is 'euclidean'.
            algorithm (str): The algorithm to compute the MST. Options include 'kruskal' and 'prim'. Default is 'kruskal'.
            handle_disconnected (bool): Whether to handle disconnected components by connecting them with epsilon-weighted edges. Default is False.
            epsilon (float): The weight to assign to connecting edges when handling disconnected components. Default is 1e-5.

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
            handle_disconnected (bool): Whether to handle disconnected components by connecting them with epsilon-weighted edges. Default is True.
            epsilon (float): The weight to assign to connecting edges when handling disconnected components. Default is 1e-5.

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
