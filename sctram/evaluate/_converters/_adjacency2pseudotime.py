#!/usr/bin/env python3

from typing import Optional, Literal

import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csgraph, issparse, diags
from scipy.sparse.linalg import eigsh
from functools import cached_property


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
        self._converted_to_distance = False

    def _ensure_distance_weights(self):
        """Ensures that the adjacency matrix represents distances.

        If the adjacency matrix represents similarities, it transforms them into distances.
        This is a private method used internally to prepare the graph for algorithms that require distance metrics.

        Raises:
            ValueError: When adjacency matrix contains negative weights, which are invalid for distance metrics.
        """
        if self._converted_to_distance:
            raise RuntimeError("`__ensure_distance_weights` is called twice.")
        
        if np.allclose(self.adjacency_matrix, self.adjacency_matrix.T):
            # Assuming similarities; convert to distances
            max_weight = self.adjacency_matrix.max()
            self.adjacency_matrix = max_weight - self.adjacency_matrix
            np.fill_diagonal(self.adjacency_matrix, 0)
            self.graph = nx.from_numpy_array(self.adjacency_matrix)
        else:
            raise ValueError("Adjacency matrix is not symetrical.")
        
        if np.any(self.adjacency_matrix < 0):  # Ensure non-negativity
            raise ValueError("Adjacency matrix contains negative weights, which are invalid for distance metrics.")
            
        self._converted_to_distance = True

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

    def _transition_matrix(self):
        """Computes and caches the transition matrix."""
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        if np.any(degrees == 0):
            raise ValueError("Graph contains isolated nodes.")

        D_inv = diags(1.0 / degrees)
        transition_matrix = D_inv @ self.adjacency_matrix

        return transition_matrix

    def diffusion_components(self, n_steps: int):
        """Computes the eigenvalues and eigenvectors of the diffusion operator derived from the adjacency matrix.
        
        The diffusion operator models the spread of a signal (such as information or influence) through the
        network, mimicking a random walk process. This method raises the transition matrix to the power of 
        `n_steps` to simulate diffusion over time. It then performs eigen decomposition on the resulting 
        diffusion operator to extract the principal components of diffusion.

        Parameters:
        - n_steps (int): The number of steps over which to simulate the diffusion. If n_steps is 0, the 
        eigenvalues and eigenvectors of the initial transition matrix are computed.

        Returns:
        - tuple: A tuple containing two elements:
            1. eigvals (numpy.ndarray): An array of eigenvalues of the diffusion operator.
            2. eigvecs (numpy.ndarray): A 2D array where each column is an eigenvector corresponding to 
            an eigenvalue in `eigvals`.

        Raises:
        - RuntimeError: If the eigen decomposition fails to converge, indicating an issue with numerical
        stability or matrix properties."""
        
        # Compute the diffusion operator
        if n_steps > 0:
            diffusion_operator = self._transition_matrix() ** n_steps
        else:
            diffusion_operator = self._transition_matrix()

        # Compute the eigenvalues and eigenvectors
        n_components = min(self.adjacency_matrix.shape[0] - 1, 50)  # Adjust 50 as needed
        try:
            if issparse(diffusion_operator):
                eigvals, eigvecs = eigsh(diffusion_operator, k=n_components, which="LM")
            else:
                eigvals, eigvecs = np.linalg.eig(diffusion_operator)
                idx = np.argsort(-np.abs(eigvals))
                eigvals = eigvals[idx][:n_components]
                eigvecs = eigvecs[:, idx][:, :n_components]
        except Exception as e:
            raise RuntimeError("Eigen decomposition did not converge.") from e

        return eigvals, eigvecs

    def to_diffusion_pseudotime_with_eigen(
        self,
        root_cell: int = 0,
        n_components: int = 100,
        n_steps: int = 100,
    ) -> np.ndarray:
        """Converts the adjacency matrix into a pseudotime array using the Diffusion-Based method.

        This method leverages the spectral properties of the diffusion operator derived from the
        adjacency matrix to compute pseudotime values. It performs eigen decomposition to identify
        principal diffusion components and assigns pseudotime based on the distance of each cell
        from the root cell in the diffusion space.

        Mathematical Formulation:
            1.  Compute the transition matrix P from the adjacency matrix A.
            2.  Raise the transition matrix to the power of `n_steps` to model diffusion over multiple steps:
                P_diffusion = P^n_steps
            3.  Perform eigen decomposition on P_diffusion to obtain eigenvalues and eigenvectors.
            4.  Select the top `n_components` eigenvectors (excluding the first trivial component).
            5.  Project each cell onto the selected diffusion components.
            6.  Calculate the Euclidean distance of each cell from the root cell in the diffusion space.
            7.  Normalize the distances to range between 0 and 1 to obtain pseudotime values.

        Args:
            root_cell (int, optional): The index of the root cell from which pseudotime is calculated.
                Default is 0.
            n_components (int, optional): The number of diffusion components (eigenvectors) to consider
                for dimensionality reduction. Higher values capture more diffusion dynamics but may
                introduce noise. Default is 10.
            n_steps (int, optional): The number of diffusion steps to simulate. A higher number allows
                the diffusion process to capture more global structures in the graph. If set to 0,
                the transition matrix is used as is without raising to any power. Default is 0.

        Returns:
            np.ndarray: A one-dimensional array of normalized pseudotime values for each cell,
            ranging from 0 to 1, where 0 corresponds to the root cell and 1 represents the most
            diffused cells.

        Raises:
            ValueError: If `root_cell` is out of bounds, if `n_components` exceeds the number of available
                eigenvectors, or if the adjacency matrix contains isolated nodes.
            RuntimeError: If eigen decomposition fails to converge.
        """
        if root_cell < 0 or root_cell >= self.adjacency_matrix.shape[0]:
            raise ValueError(f"root_cell must be between 0 and {self.adjacency_matrix.shape[0] - 1}.")

        # Compute eigenvalues and eigenvectors for the diffusion operator with given n_steps
        eigvals, eigvecs = self.diffusion_components(n_steps)

        # Adjust the number of components
        n_components = min(n_components + 1, eigvecs.shape[1])
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

        # Exclude the first trivial component
        diff_map = eigvecs[:, 1:]

        # Compute distances in diffusion space from the root cell
        diff_dist = np.linalg.norm(diff_map - diff_map[root_cell], axis=1)
        pseudotime = diff_dist / np.max(diff_dist)

        return pseudotime

    def to_diffusion_pseudotime_with_damping(
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

        return 1 - f

    def _laplacian_eigendecomposition(self):
        """Computes and caches the eigenvalues and eigenvectors of the Laplacian."""
        # Compute the graph Laplacian
        laplacian = csgraph.laplacian(self.adjacency_matrix, normed=True)

        # Ensure symmetry
        if not np.allclose(laplacian, laplacian.T):
            raise RuntimeError("Unexpected asymetry.")
            # laplacian = (laplacian + laplacian.T) / 2

        # Compute the eigenvalues and eigenvectors
        try:
            if issparse(laplacian):
                eigvals, eigvecs = eigsh(laplacian, k=min(50, self.n_cells - 2), which="SM")
            else:
                eigvals, eigvecs = eigh(laplacian)
        except Exception as e:
            raise RuntimeError("Eigen decomposition did not converge.") from e

        return eigvals, eigvecs

    def _normalize_pseudotime(self, pseudotime: np.ndarray) -> np.ndarray:
        finite_mask = np.isfinite(pseudotime)
        min_val = pseudotime[finite_mask].min()
        max_val = pseudotime[finite_mask].max()

        if max_val - min_val == 0:
            raise ValueError("Pseudotime has zero variance; cannot normalize.")

        normalized_pseudotime = (pseudotime - min_val) / (max_val - min_val)
        normalized_pseudotime[~finite_mask] = np.nan  # Assign NaN to infinite values

        return normalized_pseudotime

    def to_spectral_pseudotime(self, n_components: int = 2, root_cell: Optional[int] = None) -> np.ndarray:
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

        eigenvalues, eigenvectors = self._laplacian_eigendecomposition()

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

        pseudotime = self._normalize_pseudotime(fiedler_vector)

        if root_cell is not None:
            # Orient the pseudotime based on the root cell
            direction = pseudotime[root_cell]
            if direction > 0.5:
                pseudotime = 1 - pseudotime  # Flip the direction

        return pseudotime


class LabelAdjacencyPseudotimeConverter:
    """A class to convert a label-level adjacency matrix into a cell-level pseudotime array.

    This involves two main steps:
        1. Compute pseudotime for each label using graph-based methods.
        2. Assign pseudotime to each cell based on its label's pseudotime.

    Supported Methods for Pseudotime Computation:
        - Shortest Path-Based Pseudotime
        - Diffusion-Based Pseudotime
        - Spectral Ordering Pseudotime

    Attributes:
        label_adjacency_matrix (np.ndarray): The input label-level adjacency matrix (l x l).
        cell_labels (np.ndarray): Array mapping each cell to its corresponding label.
        label_pseudotime (Optional[np.ndarray]): The resulting pseudotime array for labels.
        cell_pseudotime (Optional[np.ndarray]): The resulting pseudotime array for cells.
    """

    def __init__(self, label_adjacency_matrix: np.ndarray, label_adjacency_matrix_labels: np.ndarray, cell_labels: np.ndarray):
        """Initializes the converter with a label adjacency matrix and cell labels.

        Args:
            label_adjacency_matrix (np.ndarray): A two-dimensional binary or weighted adjacency matrix (l x l).
            label_adjacency_matrix_labels (np.ndarray): A one-dimensional array of labels for each label (l,).
            cell_labels (np.ndarray): A one-dimensional array of labels for each cell (n,).

        Raises:
            ValueError: If dimensions of the adjacency matrix or cell_labels are invalid.
            TypeError: If `label_adjacency_matrix` or `cell_labels` are not numpy arrays.
        """
        if not isinstance(label_adjacency_matrix, np.ndarray):
            raise TypeError("label_adjacency_matrix must be a numpy array.")
        if label_adjacency_matrix.ndim != 2 or label_adjacency_matrix.shape[0] != label_adjacency_matrix.shape[1]:
            raise ValueError("label_adjacency_matrix must be a square two-dimensional array.")
        if not isinstance(label_adjacency_matrix_labels, np.ndarray) or not isinstance(cell_labels, np.ndarray):
            raise TypeError("Labels must be a numpy array.")
        if label_adjacency_matrix_labels.ndim != 1 or cell_labels.ndim != 1:
            raise ValueError("Labels must be a one-dimensional array.")
        
        unique_adj_labels = np.unique(label_adjacency_matrix_labels)
        unique_labels = np.unique(cell_labels)
        if label_adjacency_matrix.shape[0] != len(unique_adj_labels):
            raise ValueError("Unique cell_labels do not match the dimensions of adjacency matrix.")
        if len(unique_adj_labels) != len(label_adjacency_matrix_labels):
            raise ValueError(f"Adjacency labels are not unique.")
        self.label_adjacency_matrix = label_adjacency_matrix
        self.cell_labels = cell_labels
        self.adj_labels = label_adjacency_matrix_labels
        
        self.label_pseudotime: Optional[np.ndarray] = None
        self.cell_pseudotime: Optional[np.ndarray] = None

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
            root_label = kwargs.get("root_label", 0)  # TODO: create warning if it is chosen the default.
            self.label_pseudotime = converter.to_shortest_path_pseudotime(
                root_cell=root_label,
                weight="weight",
                handle_disconnected=handle_disconnected,
                alternative_distance=alternative_distance,
            )
        elif method == "diffusion_with_damping":
            alpha = kwargs.get("alpha", 0.5)
            n_steps = kwargs.get("n_steps", 100)
            tol = kwargs.get("tol", 1e-6)
            root_label = kwargs.get("root_label", 0)
            self.label_pseudotime = converter.to_diffusion_pseudotime_with_damping(
                alpha=alpha, n_steps=n_steps, tol=tol, root_cell=root_label
            )
        elif method == "diffusion_with_eigen":
            n_steps = kwargs.get("n_steps", 100)
            root_label = kwargs.get("root_label", 0)
            root_label = kwargs.get("root_label", 0)
            n_components = kwargs.get("n_components", 100)
            self.label_pseudotime = converter.to_diffusion_pseudotime_with_eigen(
                n_steps=n_steps, root_cell=root_label, n_components=n_components
            )
        elif method == "spectral":
            n_components = kwargs.get("n_components", 2)
            root_label = kwargs.get("root_label", None)
            self.label_pseudotime = converter.to_spectral_pseudotime(
                n_components=n_components, root_cell=root_label
            )
        else:
            raise ValueError(f"Unsupported method {method!r}.")

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
        label_to_pseudotime = {label: self.label_pseudotime[idx] for idx, label in enumerate(self.adj_labels)}

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
