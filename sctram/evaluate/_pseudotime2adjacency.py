#!/usr/bin/env python3

from typing import Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.sparse import csgraph
from scipy.linalg import eigh


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

        Args
            adjacency_matrix (np.ndarray): A two-dimensional binary or weighted adjacency matrix.
        
        Raises:
            ValueError: If adjacency_matrix is not a two-dimensional square NumPy array.
        """
        if not isinstance(adjacency_matrix, np.ndarray):
            raise TypeError("adjacency_matrix must be a NumPy array.")
        if adjacency_matrix.ndim != 2:
            raise ValueError("adjacency_matrix must be a two-dimensional array.")
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be a square matrix.")
        
        self.adjacency_matrix = adjacency_matrix
        self.graph = nx.from_numpy_array(adjacency_matrix)

    def to_shortest_path_pseudotime(self, root_cell: int = 0, weight: Optional[str] = 'weight') -> np.ndarray:
        """Converts the adjacency matrix into a pseudotime array using the Shortest Path-Based method.

        In this method, pseudotime values are assigned based on the shortest path lengths
        from a specified root cell to all other cells in the graph.

        Mathematical Formulation:
            1.  Let G = (V, E) be the graph where V is the set of cells and E is the set of edges.
            2.  Choose a root cell r ∈ V.
            3.  Compute the shortest path length d(r, v) for each v ∈ V using Dijkstra's algorithm.
            4.  Assign pseudotime(v) = d(r, v).
        
        Args
            root_cell (int, optional): The index of the root cell from which pseudotime is calculated. Default is 0.
            weight (Optional[str], optional): Edge attribute to use as weight. 
                If None, unweighted shortest paths are computed. Default is 'weight'.
        
        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.
        
        Raises:
            ValueError: If root_cell is not a valid node index.
        """
        if root_cell < 0 or root_cell >= self.adjacency_matrix.shape[0]:
            raise ValueError(f"root_cell must be between 0 and {self.adjacency_matrix.shape[0] - 1}.")

        try:
            lengths = nx.single_source_dijkstra_path_length(self.graph, root_cell, weight=weight)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path found from root_cell {root_cell} to other cells.")

        # Initialize pseudotime with infinities
        pseudotime = np.full(self.adjacency_matrix.shape[0], np.inf)
        
        # Assign computed lengths
        for cell, length in lengths.items():
            pseudotime[cell] = length
        
        # Handle disconnected cells by assigning the maximum finite pseudotime plus one
        max_finite = np.max(pseudotime[np.isfinite(pseudotime)])
        pseudotime[np.isinf(pseudotime)] = max_finite + 1

        return pseudotime

    def to_diffusion_pseudotime(self, alpha: float = 0.5, n_steps: int = 10, root_cell: int = 0) -> np.ndarray:
        """
        Converts the adjacency matrix into a pseudotime array using the Diffusion-Based method.

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
        
        Args
            alpha (float, optional): Damping factor controlling the influence of the diffusion. 
                Must be between 0 and 1. Default is 0.5.
            n_steps (int, optional): Maximum number of diffusion steps. Default is 10.
            root_cell (int, optional): The index of the root cell from which diffusion starts. Default is 0.
        
        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.
        
        Raises:
            ValueError: If alpha is not between 0 and 1, or if root_cell is invalid.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1.")
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

        # Iteratively compute F = alpha * P^T F + (1 - alpha) e_r
        e_r = np.zeros(self.adjacency_matrix.shape[0])
        e_r[root_cell] = 1.0

        for step in range(n_steps):
            F_new = alpha * p.T.dot(f) + (1 - alpha) * e_r
            # Check for convergence (optional: could set a tolerance)
            if np.allclose(f, F_new, atol=1e-6):
                break
            f = F_new

        return f

    def to_spectral_pseudotime(self, n_components: int = 2, root_cell: Optional[int] = None) -> np.ndarray:
        """
        Converts the adjacency matrix into a pseudotime array using the Spectral Ordering method.

        This method utilizes spectral embedding by computing the eigenvectors of the graph Laplacian
        and assigns pseudotime based on the projection of cells onto the principal eigenvector.

        Mathematical Formulation:
            1.  Let G = (V, E) be the graph with adjacency matrix A.
            2.  Compute the unnormalized graph Laplacian L = D - A, where D is the degree matrix.
            3.  Compute the first non-trivial eigenvector (Fiedler vector) of L.
            4.  Assign pseudotime(v) based on the values of the Fiedler vector.
        
        Args
            n_components (int, optional): Number of eigenvectors to compute. Default is 2.
            root_cell (Optional[int], optional): The index of the root cell to orient the pseudotime. If None, 
                pseudotime is assigned based on the Fiedler vector. Default is None.
        
        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.
        
        Raises:
            ValueError: If n_components is not a positive integer, or if root_cell is invalid.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        if root_cell is not None:
            if root_cell < 0 or root_cell >= self.adjacency_matrix.shape[0]:
                raise ValueError(f"root_cell must be between 0 and {self.adjacency_matrix.shape[0] - 1}.")

        # Compute the unnormalized Laplacian
        laplacian = csgraph.laplacian(self.adjacency_matrix, normed=False)

        # Compute the first few eigenvectors
        eigenvalues, eigenvectors = eigh(laplacian)

        # The first eigenvalue is zero (for connected graphs); the second is the Fiedler vector
        if n_components > len(eigenvalues):
            raise ValueError(f"n_components cannot exceed the number of eigenvalues ({len(eigenvalues)}).")

        # Select the second smallest eigenvector as the Fiedler vector
        if n_components >= 2:
            fiedler_vector = eigenvectors[:, 1]
        else:
            fiedler_vector = eigenvectors[:, 0]  # First eigenvector (all ones) if n_components=1

        # Normalize the Fiedler vector to range [0, 1]
        fiedler_norm = (fiedler_vector - fiedler_vector.min()) / (fiedler_vector.max() - fiedler_vector.min())

        if root_cell is not None:
            # Orient the pseudotime based on the root cell
            direction = fiedler_norm[root_cell]
            if direction < 0.5:
                fiedler_norm = 1 - fiedler_norm  # Flip the direction

        pseudotime = fiedler_norm
        return pseudotime


class PseudotimeAdjacencyConverter:
    """
    A class to convert a pseudotime array into an adjacency matrix using various methods.

    Pseudotime analysis assigns a scalar value to each cell, representing its position along a
    biological trajectory. Converting pseudotime values into an adjacency matrix enables the construction
    of graph-based representations of cellular trajectories, facilitating downstream analyses such as
    trajectory inference, clustering, and network analysis.

    Supported Methods:
        - k-Nearest Neighbors (k-NN) based adjacency
        - Threshold-based adjacency
        - Minimum Spanning Tree (MST) based adjacency

    Attributes:
        pseudotime (np.ndarray): Array of pseudotime values for each cell.
        adjacency_matrix (Optional[np.ndarray]): The resulting adjacency matrix after conversion.
    """

    def __init__(self, pseudotime: np.ndarray):
        """
        Initializes the PseudotimeAdjacencyConverter with a pseudotime array.

        Args
            pseudotime (np.ndarray): A one-dimensional array of pseudotime values.
        
        Raises:
            ValueError: If pseudotime is not a one-dimensional NumPy array.
        """
        if not isinstance(pseudotime, np.ndarray):
            raise TypeError("pseudotime must be a NumPy array.")
        if pseudotime.ndim != 1:
            raise ValueError("pseudotime must be a one-dimensional array.")
        
        self.pseudotime = pseudotime

    def to_knn_adjacency(self, n_neighbors: int = 5, metric: str = 'euclidean', 
                        algorithm: str = 'auto') -> np.ndarray:
        """
        Converts the pseudotime array into an adjacency matrix using the k-Nearest Neighbors (k-NN) method.

        In this method, each cell is connected to its k nearest neighbors based on the pseudotime values.
        The distance metric can be specified, with 'euclidean' being the default.

        Mathematical Formulation:
            1.  Let t_i be the pseudotime of cell i.
            2.  The distance between cells i and j is defined as |t_i - t_j|.
            3.  For each cell i, identify the k cells with the smallest distances to i.
            4.  Set A_{i,j} = 1 if cell j is among the k nearest neighbors of cell i; otherwise, set A_{i,j} = 0.
            5.  The adjacency matrix A is symmetric, meaning if A_{i,j} = 1, then A_{j,i} = 1.
            
        Process:
            1.	Reshapes the pseudotime array to a two-dimensional format suitable for NearestNeighbors.
            2.	Initializes the NearestNeighbors model with the specified parameters.
            3.	Computes the nearest neighbors for each cell, including itself.
            4.	Constructs a binary adjacency matrix where A[i, j] = 1 if cell j is among the k nearest 
                neighbors of cell i, ensuring symmetry.

        Args
            n_neighbors (int, optional): Number of nearest neighbors to connect. Default is 5.
            metric (str, optional): Distance metric to use. Default is 'euclidean'.
            algorithm (str, optional): Algorithm to compute nearest neighbors. Default is 'auto'.
        
        Returns:
            np.ndarray: A binary adjacency matrix of shape (n_cells, n_cells).
        
        Raises:
            ValueError: If n_neighbors is not a positive integer.
        """
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        
        # Reshape pseudotime for k-NN computation
        pseudotime_reshaped = self.pseudotime.reshape(-1, 1)
        
        # Initialize NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, 
                                metric=metric, 
                                algorithm=algorithm).fit(pseudotime_reshaped)
        
        # Compute the k+1 nearest neighbors (including the cell itself)
        distances, indices = nbrs.kneighbors(pseudotime_reshaped)
        
        n_cells = len(self.pseudotime)
        adjacency = np.zeros((n_cells, n_cells), dtype=int)
        
        for idx, neighbors in enumerate(indices):
            # Exclude the first neighbor (the cell itself)
            for neighbor in neighbors[1:]:
                adjacency[idx, neighbor] = 1
                adjacency[neighbor, idx] = 1  # Ensure symmetry
        
        return adjacency

    def to_threshold_adjacency(self, delta: float) -> np.ndarray:
        """
        Converts the pseudotime array into an adjacency matrix by thresholding pseudotime differences.

        In this method, two cells are connected if the absolute difference in their pseudotime values
        is less than a specified threshold delta.

        Mathematical Formulation:
            1.  Let t_i and t_j be the pseudotime values of cells i and j, respectively.
            2.  Define A_{i,j} = 1 if |t_i - t_j| < delta; otherwise, A_{i,j} = 0.
            3.  The adjacency matrix A is symmetric.
            
        Process:
	        1.	Computes the absolute difference matrix between all pairs of pseudotime values.
	        2.	Applies the threshold to create a binary adjacency matrix where A[i, j] = 1 if |t_i - t_j| < delta.
	        3.	Removes self-connections by setting the diagonal to zero.

        Args
            delta (float): The maximum allowed pseudotime difference for two cells to be connected.
        
        Returns:
            np.ndarray: A binary adjacency matrix of shape (n_cells, n_cells).
        
        Raises:
            ValueError: If delta is not a positive float.
        """
        if not isinstance(delta, (int, float)) or delta <= 0:
            raise ValueError("delta must be a positive number.")
        
        n_cells = len(self.pseudotime)
        adjacency = np.zeros((n_cells, n_cells), dtype=int)
        
        # Compute the absolute difference matrix
        pseudotime_matrix = np.tile(self.pseudotime, (n_cells, 1))
        diff_matrix = np.abs(pseudotime_matrix - pseudotime_matrix.T)
        
        # Apply threshold
        adjacency[diff_matrix < delta] = 1
        
        # Remove self-connections
        np.fill_diagonal(adjacency, 0)
        
        return adjacency

    def to_mst_adjacency(self) -> np.ndarray:
        """
        Converts the pseudotime array into an adjacency matrix using the Minimum Spanning Tree (MST) method.

        In this method, an MST is constructed where each node represents a cell, and edges connect
        cells with minimal pseudotime differences, ensuring that the graph is fully connected with no cycles.

        Mathematical Formulation:
            1.  Construct a complete graph where each node represents a cell.
            2.  The weight of the edge between cell i and cell j is |t_i - t_j|.
            3.  Compute the MST of this graph using algorithms such as Kruskal's or Prim's.
            4.  Represent the MST as an adjacency matrix A, where A_{i,j} = 1 if cells i and j are connected in 
                the MST; otherwise, A_{i,j} = 0.

        Process:
            1.	Creates a complete graph where each node represents a cell, and edge weights are the absolute 
                differences in pseudotime values.
            2.	Computes the MST using Kruskal's algorithm.
            3.	Constructs a binary adjacency matrix from the MST edges, ensuring symmetry.

        Returns:
            np.ndarray: A binary adjacency matrix of shape (n_cells, n_cells) representing the MST.
        
        Notes:
            - This method ensures that all cells are connected in a tree structure.
            - The resulting adjacency matrix will have exactly (n_cells - 1) edges.
        """
        n_cells = len(self.pseudotime)
        
        # Create a complete graph with pseudotime differences as edge weights
        G = nx.Graph()
        G.add_nodes_from(range(n_cells))
        
        # Add edges with weights
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                weight = abs(self.pseudotime[i] - self.pseudotime[j])
                G.add_edge(i, j, weight=weight)
        
        # Compute the MST
        mst = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
        
        # Initialize adjacency matrix
        adjacency = np.zeros((n_cells, n_cells), dtype=int)
        
        # Populate adjacency matrix based on MST edges
        for edge in mst.edges():
            i, j = edge
            adjacency[i, j] = 1
            adjacency[j, i] = 1  # Ensure symmetry
        
        return adjacency


class LabelAdjacencyPseudotimeConverter:
    """
    A class to convert a label-level adjacency matrix into a cell-level pseudotime array.
    
    This involves two main steps:
        1. Compute pseudotime for each label using graph-based methods.
        2. Assign pseudotime to each cell based on its label's pseudotime.
    
    Attributes:
        label_adjacency_matrix (np.ndarray): The input label-level adjacency matrix (l x l).
        label_graph (networkx.Graph): The graph constructed from the label adjacency matrix.
        label_pseudotime (Optional[np.ndarray]): The resulting pseudotime array for labels.
        cell_labels (np.ndarray): Array mapping each cell to its corresponding label.
        cell_pseudotime (Optional[np.ndarray]): The resulting pseudotime array for cells.
    """

    def __init__(self, label_adjacency_matrix: np.ndarray, cell_labels: np.ndarray):
        """
        Initializes the converter with a label adjacency matrix and cell labels.
        
        Args
            label_adjacency_matrix (np.ndarray): A two-dimensional binary or weighted adjacency matrix (l x l).
            cell_labels (np.ndarray): A one-dimensional array of labels for each cell (n,).
        
        Raises:
            ValueError: If dimensions of the adjacency matrix or cell_labels are invalid.
        """
        if not isinstance(label_adjacency_matrix, np.ndarray):
            raise TypeError("label_adjacency_matrix must be a NumPy array.")
        if label_adjacency_matrix.ndim != 2 or label_adjacency_matrix.shape[0] != label_adjacency_matrix.shape[1]:
            raise ValueError("label_adjacency_matrix must be a square two-dimensional array.")
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a NumPy array.")
        if cell_labels.ndim != 1:
            raise ValueError("cell_labels must be a one-dimensional array.")
        
        self.label_adjacency_matrix = label_adjacency_matrix
        self.label_graph = nx.from_numpy_array(label_adjacency_matrix)
        self.cell_labels = cell_labels
        self.label_pseudotime = None
        self.cell_pseudotime = None

    def compute_label_pseudotime(self, method: str = 'shortest_path', **kwargs) -> np.ndarray:
        """
        Computes pseudotime for each label using the specified method.
        
        Supported Methods:
            - 'shortest_path': Uses shortest path-based pseudotime.
            - 'diffusion': Uses diffusion-based pseudotime.
            - 'spectral': Uses spectral ordering pseudotime.
        
        Args
            method (str, optional): The method to compute pseudotime. Default is 'shortest_path'.
            **kwargs: Additional keyword arguments for the chosen method.
        
        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each label.
        
        Raises:
            ValueError: If an unsupported method is specified.
        """
        if method == 'shortest_path':
            root_label = kwargs.get('root_label', 0)
            converter = AdjacencyPseudotimeConverter(self.label_adjacency_matrix)
            self.label_pseudotime = converter.to_shortest_path_pseudotime(root_cell=root_label, weight='weight')
        elif method == 'diffusion':
            alpha = kwargs.get('alpha', 0.5)
            n_steps = kwargs.get('n_steps', 10)
            root_label = kwargs.get('root_label', 0)
            converter = AdjacencyPseudotimeConverter(self.label_adjacency_matrix)
            self.label_pseudotime = converter.to_diffusion_pseudotime(alpha=alpha, n_steps=n_steps, root_cell=root_label)
        elif method == 'spectral':
            n_components = kwargs.get('n_components', 2)
            root_label = kwargs.get('root_label', None)
            converter = AdjacencyPseudotimeConverter(self.label_adjacency_matrix)
            self.label_pseudotime = converter.to_spectral_pseudotime(n_components=n_components, root_cell=root_label)
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose from 'shortest_path', 'diffusion', 'spectral'.")
        
        return self.label_pseudotime

    def assign_cell_pseudotime(self) -> np.ndarray:
        """
        Assigns pseudotime to each cell based on its label's pseudotime.
        If label pseudotime has not been computed, it raises an error.
        
        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.
        
        Raises:
            ValueError: If label_pseudotime has not been computed.
        """
        if self.label_pseudotime is None:
            raise ValueError("Label pseudotime has not been computed. Call compute_label_pseudotime() first.")
        
        # Map each cell to its label's pseudotime
        self.cell_pseudotime = self.label_pseudotime[self.cell_labels]
        return self.cell_pseudotime

    def get_cell_pseudotime(self, method: str = 'shortest_path', **kwargs) -> np.ndarray:
        """
        Convenience method to compute label pseudotime and assign it to cells.
        
        Args
            method (str, optional): The method to compute pseudotime. Default is 'shortest_path'.
            **kwargs: Additional keyword arguments for the chosen method.
        
        Returns:
            np.ndarray: A one-dimensional array of pseudotime values for each cell.
        """
        self.compute_label_pseudotime(method=method, **kwargs)
        return self.assign_cell_pseudotime()


class PseudotimeLabelAdjacencyConverter:
    """
    A class to convert a cell-level pseudotime array into a label-level adjacency matrix.
    
    This involves aggregating pseudotime values per label and then constructing an adjacency matrix
    based on the relationships between labels derived from cell pseudotime.
    
    Supported Methods for Aggregation:
        - 'mean': Average pseudotime per label.
        - 'median': Median pseudotime per label.
        - 'min': Minimum pseudotime per label.
        - 'max': Maximum pseudotime per label.
    
    Supported Methods for Adjacency Construction:
        - 'knn': k-Nearest Neighbors based adjacency.
        - 'threshold': Threshold-based adjacency.
        - 'mst': Minimum Spanning Tree based adjacency.
    
    Attributes:
        cell_pseudotime (np.ndarray): Array of pseudotime values for each cell (n x 1).
        cell_labels (np.ndarray): Array mapping each cell to its corresponding label (n,).
        label_adjacency_matrix (Optional[np.ndarray]): The resulting label-level adjacency matrix (l x l).
    """

    def __init__(self, cell_pseudotime: np.ndarray, cell_labels: np.ndarray):
        """
        Initializes the converter with a cell pseudotime array and cell labels.
        
        Args
            cell_pseudotime (np.ndarray): A one-dimensional array of pseudotime values for each cell (n,).
            cell_labels (np.ndarray): A one-dimensional array of labels for each cell (n,).
        
        Raises:
            ValueError: If dimensions of the inputs are invalid or if labels are inconsistent.
        """
        if not isinstance(cell_pseudotime, np.ndarray):
            raise TypeError("cell_pseudotime must be a NumPy array.")
        if cell_pseudotime.ndim != 1:
            raise ValueError("cell_pseudotime must be a one-dimensional array.")
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a NumPy array.")
        if cell_labels.ndim != 1:
            raise ValueError("cell_labels must be a one-dimensional array.")
        if len(cell_pseudotime) != len(cell_labels):
            raise ValueError("cell_pseudotime and cell_labels must have the same length.")
        
        self.cell_pseudotime = cell_pseudotime
        self.cell_labels = cell_labels
        self.label_adjacency_matrix = None

    def aggregate_pseudotime_per_label(self, aggregation: str = 'mean') -> np.ndarray:
        """
        Aggregates pseudotime values per label using the specified method.
        
        Supported Aggregation Methods:
            - 'mean': Average pseudotime per label.
            - 'median': Median pseudotime per label.
            - 'min': Minimum pseudotime per label.
            - 'max': Maximum pseudotime per label.
        
        Args
            aggregation (str, optional): The aggregation method to use. Default is 'mean'.
        
        Returns:
            np.ndarray: A one-dimensional array of aggregated pseudotime values per label (l,).
        
        Raises:
            ValueError: If an unsupported aggregation method is specified.
        """
        labels = np.unique(self.cell_labels)
        l = len(labels)
        aggregated_pseudotime = np.zeros(l)
        
        for idx, label in enumerate(labels):
            label_pseudotime = self.cell_pseudotime[self.cell_labels == label]
            if aggregation == 'mean':
                aggregated_pseudotime[idx] = label_pseudotime.mean()
            elif aggregation == 'median':
                aggregated_pseudotime[idx] = np.median(label_pseudotime)
            elif aggregation == 'min':
                aggregated_pseudotime[idx] = label_pseudotime.min()
            elif aggregation == 'max':
                aggregated_pseudotime[idx] = label_pseudotime.max()
            else:
                raise ValueError(f"Unsupported aggregation method '{aggregation}'. Choose from 'mean', 'median', 'min', 'max'.")
        
        self.labels = labels
        self.aggregated_pseudotime = aggregated_pseudotime
        return self.aggregated_pseudotime

    def construct_label_adjacency(self, method: str = 'knn', **kwargs) -> np.ndarray:
        """
        Constructs a label-level adjacency matrix based on aggregated pseudotime values.
        
        Supported Methods:
            - 'knn': k-Nearest Neighbors based adjacency.
            - 'threshold': Threshold-based adjacency.
            - 'mst': Minimum Spanning Tree based adjacency.
        
        Args
            method (str, optional): The method to construct adjacency. Default is 'knn'.
            **kwargs: Additional keyword arguments for the chosen method.
        
        Returns:
            np.ndarray: A binary adjacency matrix of shape (l x l).
        
        Raises:
            ValueError: If an unsupported method is specified.
        """
        if not hasattr(self, 'aggregated_pseudotime'):
            raise ValueError("Aggregated pseudotime per label has not been computed. Call aggregate_pseudotime_per_label() first.")
        
        # Initialize PseudotimeAdjacencyConverter with label pseudotime
        converter = PseudotimeAdjacencyConverter(pseudotime=self.aggregated_pseudotime)
        
        if method == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 2)
            label_adjacency = converter.to_knn_adjacency(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
        elif method == 'threshold':
            delta = kwargs.get('delta', 1.0)
            label_adjacency = converter.to_threshold_adjacency(delta=delta)
        elif method == 'mst':
            label_adjacency = converter.to_mst_adjacency()
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose from 'knn', 'threshold', 'mst'.")
        
        self.label_adjacency_matrix = label_adjacency
        return self.label_adjacency_matrix

    def get_label_adjacency(self, aggregation: str = 'mean', method: str = 'knn', **kwargs) -> np.ndarray:
        """
        Convenience method to aggregate pseudotime and construct label adjacency.
        
        Args
            aggregation (str, optional): The aggregation method to use. Default is 'mean'.
            method (str, optional): The method to construct adjacency. Default is 'knn'.
            **kwargs: Additional keyword arguments for aggregation and adjacency methods.
        
        Returns:
            np.ndarray: A binary adjacency matrix of shape (l x l).
        """
        self.aggregate_pseudotime_per_label(aggregation=aggregation)
        return self.construct_label_adjacency(method=method, **kwargs)