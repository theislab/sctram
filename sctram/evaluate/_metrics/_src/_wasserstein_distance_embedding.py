#!/usr/bin/env python3

import numpy as np
import networkx as nx
from scipy import sparse
from ot import emd2
from loguru import logger

try:
    from sctram.evaluate._metrics._src.validators import validate_zero_or_positive as _validator
    from sctram.evaluate._metrics._src.utils import convert_scanpy_neighbors_to_indices
except ImportError:
    from validators import validate_zero_or_positive as _validator
    from utils import convert_scanpy_neighbors_to_indices


_logger = logger.bind(name="MetricBase")


def wasserstein_distance_embedding(
    given_graph: nx.DiGraph,
    labels_array: np.ndarray,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    n_neighbors: int,
    validate_result: bool = True,
) -> float:
    """Wasserstein Distance (WD) between graph and embedding cell type distributions.

    Computes the average Wasserstein distance between the transition distributions
    defined by the graph and the neighbor distributions in the embedding for each cell type.

    Parameters:
        given_graph (nx.DiGraph): Biological graph with cell types as nodes and transitions as edges.
        labels_array (np.ndarray): 1D array (n_cells) of cell type labels.
        precomputed_embedded_connectivities (sparse.csr_matrix): Scanpy KNN connectivities matrix (includes self as first neighbor).
        n_neighbors (int): Number of neighbors (including self). Must match precomputed_embedded_connectivities.
        validate_result (bool): Validate the resulting score is non-negative.

    Returns:
        float: Average Wasserstein distance across all cell types. Lower values indicate better preservation.

    Statistical Intuition:
        Models the alignment between biological graph structure and embedding neighborhoods using OT theory:
        1. Graph as Railway Blueprint: 
            - Nodes = Cities (cell types), Edges = Allowed tracks (biological transitions)
            - Shortest path distances define transition costs (direct vs multi-step journeys)
        2. Embedding as Passenger Flow: 
            - KNN neighborhoods = Observed travel patterns
            - Empirical distribution shows actual cell-type transitions
        3. Wasserstein as Rerouting Cost:
            - Measures minimal "work" (Σ massxdistance) to reshape:
                - Q (embedding's observed flow) → P (graph's prescribed routes)
            - Penalizes mismatches proportionally to their topological distance in the graph
            - Accounts for both transition existence (edge presence) and validity (path length)
        4. Biological Robustness:
            - Normalized distributions handle cluster density variations
            - Orphan nodes (no outgoing edges) treated as terminal states
            - Sparse transitions preserved through cost-aware mass transport

    """
    embedded_neighbors = convert_scanpy_neighbors_to_indices(
        scanpy_neighbors_matrix=precomputed_embedded_connectivities,
        k=n_neighbors - 1  # exclude self
    )
    n_cells = embedded_neighbors.shape[0]

    # Validate input shapes
    assert embedded_neighbors.shape == (n_cells, n_neighbors - 1), \
        f"Neighbor indices shape {embedded_neighbors.shape} != ({n_cells}, {n_neighbors-1})"
    unique_labels = list(given_graph.nodes())
    t = len(unique_labels)
    if t == 0:
        raise ValueError("Given graph has no nodes.")

    # Create label to index mapping
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    # Precompute cost matrix (shortest path distances between all node pairs)
    cost_matrix = np.zeros((t, t))
    for i, u in enumerate(unique_labels):
        lengths = nx.shortest_path_length(given_graph, source=u)
        for j, v in enumerate(unique_labels):
            if v in lengths:
                cost_matrix[i, j] = lengths[v]
            else:
                cost_matrix[i, j] = t + 1  # Large penalty for unreachable nodes
        cost_matrix[i, i] = 0  # Distance to self is zero

    # For each cell type, compute WD between graph transitions and embedding neighbors
    wd_scores = []
    for u in unique_labels:
        # Get indices of cells of type u
        cells_u = np.where(labels_array == u)[0]
        if len(cells_u) == 0:
            raise ValueError(f"No cells found for label {u}. Skipping.")

        # Extract neighbor labels for these cells
        u_neighbor_indices = embedded_neighbors[cells_u, :].flatten()
        u_neighbor_labels = labels_array[u_neighbor_indices]

        # Compute Q distribution (embedding neighbors)
        unique_q, counts_q = np.unique(u_neighbor_labels, return_counts=True)
        
        # Filtering step to handle unknown labels
        valid_mask = np.isin(unique_q, unique_labels)
        valid_unique_q = unique_q[valid_mask]
        valid_counts_q = counts_q[valid_mask]

        q = np.zeros(t)
        for lbl, cnt in zip(valid_unique_q, valid_counts_q):
            q[label_to_idx[lbl]] += cnt
        q_sum = q.sum()
        if q_sum == 0:
            # No valid neighbors for label {u}. Using uniform Q
            q = np.ones(t) / t
        else:
            q /= q_sum

        # Compute P distribution (graph transitions)
        graph_neighbors = list(given_graph.neighbors(u))
        if not graph_neighbors:
            # No outgoing edges: assume self-transition
            p = np.zeros(t)
            p[label_to_idx[u]] = 1.0
        else:
            # Handle weighted or unweighted graph
            if nx.is_weighted(given_graph):
                weights = np.array([given_graph[u][v].get('weight', 1.0) for v in graph_neighbors])
            else:
                weights = np.ones(len(graph_neighbors))
            weights_sum = weights.sum()
            if weights_sum <= 0:
                weights = np.ones(len(graph_neighbors))
                weights_sum = len(graph_neighbors)
            p = np.zeros(t)
            for v, w in zip(graph_neighbors, weights):
                p[label_to_idx[v]] += w / weights_sum

        # Compute Wasserstein distance
        wd = emd2(p, q, cost_matrix)

        if not np.isnan(wd):
            wd_scores.append(wd)

    if not wd_scores:
        raise ValueError("No valid WD computed. Check input data.")

    avg_wd = np.mean(wd_scores)

    if validate_result:
        _validator(score=avg_wd)
    
    return avg_wd


if __name__ == '__main__':
    
    def create_valid_scanpy_neighbors(n_cells: int, k: int, seed_row: int = 0) -> sparse.csr_matrix:
        """Create a valid scanpy neighbor matrix where:
        - seed_row has neighbors [1, 2, ..., k]
        - All other rows have cyclic neighbors to satisfy k requirements
        """
        indices = []
        indptr = [0]
        
        # Create neighbors for seed_row
        seed_neighbors = np.arange(1, k+1)
        indices.extend(seed_neighbors)
        indptr.append(len(indices))
        
        # Create neighbors for other rows (cyclically repeat valid indices)
        for i in range(1, n_cells):
            row_neighbors = np.arange(k)  # Valid cyclic indices
            indices.extend(row_neighbors)
            indptr.append(len(indices))
        
        data = np.ones(len(indices), dtype=np.float32)
        return sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
    
    def test_perfect_match():
        """Test case where graph transitions perfectly match embedding neighbors (WD=0)."""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'A')])
        
        labels = np.array(['A', 'B'])
        n_neighbors = 2
        embedded_conn = create_valid_scanpy_neighbors(n_cells=2, k=1, seed_row=0)
        
        result = wasserstein_distance_embedding(G, labels, embedded_conn, n_neighbors, validate_result=True)
        expected = 0.0
        assert np.isclose(result, expected), f"Perfect match failed. Expected {expected}, got {result}"

    def test_orphan_node_zero_wd():
        """Test orphan node (no outgoing edges) with self neighbors (WD=0)."""
        G = nx.DiGraph()
        G.add_node('A')  # Orphan node
        
        labels = np.array(['A', 'A'])
        n_neighbors = 2
        embedded_conn = create_valid_scanpy_neighbors(n_cells=2, k=1, seed_row=0)
        
        result = wasserstein_distance_embedding(G, labels, embedded_conn, n_neighbors, validate_result=True)
        expected = 0.0
        assert np.isclose(result, expected), f"Orphan node test failed. Expected {expected}, got {result}"

    def test_no_overlap_high_wd():
        """Test case with complete mismatch between graph and embedding (WD=3.0)."""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'B')])
        
        labels = np.array(['A', 'A', 'B', 'B'])
        n_neighbors = 2
        embedded_conn = create_valid_scanpy_neighbors(n_cells=4, k=1, seed_row=0)
        
        result = wasserstein_distance_embedding(G, labels, embedded_conn, n_neighbors, validate_result=True)
        expected = 3.0  # Corrected from 2.0 to 3.0 based on cost matrix analysis
        assert np.isclose(result, expected), f"No overlap test failed. Expected {expected}, got {result}"

    def test_large_manual_calculation():
        """Test with manually computed WD for a structured graph and embedding."""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'C')])
        
        # Manually create embedded connectivities for 3 cells (A, B, C)
        indices = [1, 2, 2, 0, 2, 1]
        indptr = [0, 2, 4, 6]
        data = np.ones(6, dtype=np.float32)
        embedded_conn = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        
        labels = np.array(['A', 'B', 'C'])
        n_neighbors = 3  # k=2 (excluding self)
        
        result = wasserstein_distance_embedding(G, labels, embedded_conn, n_neighbors, validate_result=True)
        expected = (0.5 + 2.0 + 2.0) / 3  # Manually computed expected average WD
        assert np.isclose(result, expected, rtol=1e-3), f"Manual calculation failed. Expected {expected}, got {result}"

    def create_knn_connectivity(n_cells, n_neighbors):
        """
        Create a dummy sparse connectivity matrix (including self-neighbor).
        Here, we simulate a simple KNN graph where each cell's neighbors are chosen randomly.
        """
        rows = []
        cols = []
        data = []
        for i in range(n_cells):
            # self-neighbor is always the first entry
            neighbors = np.random.choice(np.delete(np.arange(n_cells), i), n_neighbors - 1, replace=False)
            indices = np.concatenate(([i], neighbors))
            rows.extend([i] * n_neighbors)
            cols.extend(indices)
            data.extend([1] * n_neighbors)
        connectivity = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        return connectivity

    def test_balanced_cell_types():
        """
        Scenario 1: Fully connected graph with balanced cell types.
        Every cell type has outgoing transitions to every other type, and the embedding
        neighbor distributions roughly match the graph.
        """
        # Create a graph with 3 cell types and full connectivity
        cell_types = ['A', 'B', 'C']
        G = nx.DiGraph()
        G.add_nodes_from(cell_types)
        for u in cell_types:
            for v in cell_types:
                if u != v:
                    G.add_edge(u, v, weight=1.0)
        
        # Create a labels array: 90 cells evenly distributed among 3 types
        n_cells = 90
        labels_array = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
        
        # Create a KNN connectivity matrix (neighbors include self, so n_neighbors=6 for instance)
        n_neighbors = 6
        connectivity = create_knn_connectivity(n_cells, n_neighbors)
        
        # Compute the score
        score = wasserstein_distance_embedding(G, labels_array, connectivity, n_neighbors)
        
        assert score >= 0

    def test_imbalanced_cell_types():
        """
        Scenario 2: Imbalanced cell types.
        One cell type ('A') is rare compared to others, testing if the method handles imbalances.
        """
        cell_types = ['A', 'B', 'C']
        G = nx.DiGraph()
        G.add_nodes_from(cell_types)
        # Define transitions: more transitions from B and C, and 'A' has few connections
        G.add_edge('B', 'A', weight=0.5)
        G.add_edge('B', 'C', weight=0.5)
        G.add_edge('C', 'B', weight=1.0)
        G.add_edge('A', 'A', weight=1.0)  # 'A' only transitions to itself

        # Create a labels array with imbalance: 'A' is rare
        labels_array = np.array(['A'] * 10 + ['B'] * 45 + ['C'] * 45)
        n_cells = labels_array.shape[0]
        n_neighbors = 6
        connectivity = create_knn_connectivity(n_cells, n_neighbors)
        
        score = wasserstein_distance_embedding(G, labels_array, connectivity, n_neighbors)
        
        assert score >= 0

    def test_graph_with_orphan_nodes():
        """
        Scenario 3: Graph with orphan nodes (nodes with no outgoing edges).
        Tests if the self-transition fallback works correctly.
        """
        cell_types = ['A', 'B', 'C']
        G = nx.DiGraph()
        G.add_nodes_from(cell_types)
        # Only 'B' and 'C' have transitions; 'A' is an orphan.
        G.add_edge('B', 'C', weight=1.0)
        G.add_edge('C', 'B', weight=1.0)
        
        labels_array = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
        n_cells = labels_array.shape[0]
        n_neighbors = 6
        connectivity = create_knn_connectivity(n_cells, n_neighbors)
        
        score = wasserstein_distance_embedding(G, labels_array, connectivity, n_neighbors)
        
        assert score >= 0

    def test_weighted_transitions():
        """
        Scenario 4: Weighted transitions.
        The graph has weighted edges, and we simulate that the embedding reflects these weight differences.
        """
        cell_types = ['A', 'B', 'C', 'D']
        G = nx.DiGraph()
        G.add_nodes_from(cell_types)
        # Define a more complex weighted graph
        G.add_edge('A', 'B', weight=2.0)
        G.add_edge('A', 'C', weight=1.0)
        G.add_edge('B', 'C', weight=3.0)
        G.add_edge('C', 'D', weight=4.0)
        G.add_edge('D', 'A', weight=1.0)
        
        # Create a labels array with 120 cells in a non-uniform distribution
        labels_array = np.array(['A'] * 20 + ['B'] * 30 + ['C'] * 50 + ['D'] * 20)
        n_cells = labels_array.shape[0]
        n_neighbors = 7  # e.g., 7 neighbors including self
        connectivity = create_knn_connectivity(n_cells, n_neighbors)
        
        score = wasserstein_distance_embedding(G, labels_array, connectivity, n_neighbors)
        
        assert score >= 0

    # Execute all tests
    test_perfect_match()
    test_orphan_node_zero_wd()
    test_no_overlap_high_wd()
    test_large_manual_calculation()
    test_balanced_cell_types()
    test_imbalanced_cell_types()
    test_graph_with_orphan_nodes()
    test_weighted_transitions()
    print("All tests passed!")
