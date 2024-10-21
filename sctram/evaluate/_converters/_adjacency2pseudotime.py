#!/usr/bin/env python3

from typing import Optional

import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csgraph


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
        """Ensures that the adjacency matrix represents distances.

        If the adjacency matrix represents similarities, it transforms them into distances.
        This is a private method used internally to prepare the graph for algorithms that require distance metrics.

        Raises:
            ValueError: When adjacency matrix contains negative weights, which are invalid for distance metrics.
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
            ValueError: If `alpha` is not between 0 and 1, if `root_cell` is invalid, or
                if the adjacency matrix contains isolated nodes.
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
        p = self.adjacency_matrix / degrees[:, np.newaxis]

        # Initialize the steady-state distribution F
        f = np.zeros(self.adjacency_matrix.shape[0])
        f[root_cell] = 1.0  # Start diffusion from the root cell

        # Initialize e_r
        e_r = np.zeros(self.adjacency_matrix.shape[0])
        e_r[root_cell] = 1.0

        for step in range(n_steps):
            f_new = alpha * p.T.dot(f) + (1 - alpha) * e_r
            # Check for convergence
            if np.linalg.norm(f_new - f, ord=1) < tol:
                f = f_new
                print(f"Diffusion pseudotime converged in {step + 1} steps.")
                break
            f = f_new
        else:
            print(f"Diffusion pseudotime did not converge within {n_steps} steps.")

        return f

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
        # gaussian_graph = nx.from_numpy_array(gaussian_weights_matrix)  # TODO: this variable is not used.

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
                f"Unsupported method {method!r}. Choose from 'shortest_path', 'diffusion', 'spectral', 'gaussian_kernel'."
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
                f"Unsupported method {method!r}. Choose from 'shortest_path', "
                "'diffusion', 'spectral', 'gaussian_kernel'."
            )

        return self.label_pseudotime

    def assign_cell_pseudotime(
        self, handle_disconnected: str = "assign_max_plus_one", alternative_distance: Optional[float] = None
    ) -> np.ndarray:
        """Assigns pseudotime to each cell based on its label's pseudotime.

        Args:
            handle_disconnected (str): Strategy to handle disconnected labels. Must
                match the strategy used in `get_label_pseudotime`.
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
