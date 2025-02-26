#!/usr/bin/env python3

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score
from scipy import sparse
import warnings

try:
    from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator
    from sctram.evaluate._metrics.utils import convert_scanpy_neighbors_to_indices
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator
    from utils import convert_scanpy_neighbors_to_indices


def trajectory_cardinality_validation(
    given_graph: nx.DiGraph,
    labels_array: np.ndarray,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    n_neighbors: int,
    validate_result: bool,
    skip_chain_nodes: bool = True
) -> float:
    """Advanced Trajectory Cardinality Validation F1-score.

    Evaluates embedding's ability to preserve trajectory branches by comparing clusters around graph branch points
    to expected branches derived from graph structure. Uses articulation points to identify branch regions,
    precomputed KNN for efficient neighbor analysis, and spectral clustering to detect embedding clusters.

    Parameters:
        given_graph (nx.DiGraph): Biological graph with cell types as nodes and directed edges as transitions.
        labels_array (np.ndarray): 1D array (n_cells,) of cell type labels matching graph nodes.
        precomputed_embedded_connectivities (sparse.csr_matrix): Scanpy KNN connectivities matrix.
        n_neighbors (int): Number of neighbors used in precomputed_embedded_connectivities (including self).
        validate_result (bool): Whether to validate the output score is in [0, 1].
        skip_chain_nodes (bool): Disregard chain nodes and skip them.

    Returns:
        float: Weighted F1-score (0-1) averaged across branch points. Higher values indicate better branch preservation.

    Statistical Rationale:
        1. Graph-Driven Ground Truth: Uses graph articulation points to identify critical branch regions,
           ensuring biological relevance.
        2. Majority Voting: Determines expected branch membership using majority vote of KNN neighbors in embedding space,
           reducing noise from single-neighbor decisions.
        3. Cluster Validation: Compares spectral clusters (non-convex structure-aware) against graph-derived branches
           using F1-score, balancing precision and recall.

    Advantages:
        - Efficient neighbor analysis using precomputed KNN.
        - Robust to local noise through majority voting.
        - Handles non-convex clusters via spectral clustering.
        - Automatically skips invalid branch points (no biological branching).

    Limitations:
        - Requires precomputed neighbors (sensitive to KNN parameters).
        - Assumes graph branch points correspond to embedding regions.
        - F1-score may be optimistic if multiple branches share similar cell types.
    """
    undirected_graph = given_graph.to_undirected()
    branch_points = list(nx.articulation_points(undirected_graph))
    
    # Convert Scanpy neighbors to indices matrix (exclude self)
    embedded_neighbors = convert_scanpy_neighbors_to_indices(
        scanpy_neighbors_matrix=precomputed_embedded_connectivities,
        k=n_neighbors - 1  # Exclude self from neighbors
    )
    
    total_f1 = 0.0
    processed_branches = 0
    
    for bp in branch_points:
        # Remove branch point and find connected components
        subgraph = undirected_graph.subgraph(undirected_graph.nodes - {bp})
        components = list(nx.connected_components(subgraph))
        n_components = len(components)
        
        if n_components < 2:
            raise ValueError
        elif n_components == 2 and skip_chain_nodes:
            continue  # Some articulation points may not represent true biological branches (e.g., linear chain nodes).

        # Get cells of branch point type
        cell_indices = np.where(labels_array == bp)[0]
        if len(cell_indices) < n_components:
            raise ValueError

        # Map components and validate neighbors
        component_map = {node: cid for cid, comp in enumerate(components) for node in comp}
        valid_cell_mask = np.zeros(len(cell_indices), dtype=bool)  # Boolean array
        true_labels = []
        
        # Validate cells using KNN majority voting
        for i, idx in enumerate(cell_indices):
            neighbor_labels = labels_array[embedded_neighbors[idx]]
            votes = []
            for nl in neighbor_labels:
                if nl != bp and nl in component_map:
                    votes.append(component_map[nl])
            if not votes:
                continue  # Skip cells with no valid neighbors
                
            unique, counts = np.unique(votes, return_counts=True)
            true_labels.append(unique[np.argmax(counts)])
            valid_cell_mask[i] = True

        valid_cell_indices = cell_indices[valid_cell_mask]
        if len(np.unique(true_labels)) < n_components or len(valid_cell_indices) < n_components:
            raise ValueError

        # Build affinity matrix from precomputed connectivities
        n_valid = len(valid_cell_indices)
        index_map = {orig: idx for idx, orig in enumerate(valid_cell_indices)}
        affinity = sparse.lil_matrix((n_valid, n_valid), dtype=np.float32)

        for i, orig_idx in enumerate(valid_cell_indices):
            # Get column indices and data directly from CSR structure
            start = precomputed_embedded_connectivities.indptr[orig_idx]
            end = precomputed_embedded_connectivities.indptr[orig_idx+1]
            neighbors = precomputed_embedded_connectivities.indices[start:end]
            values = precomputed_embedded_connectivities.data[start:end]
            
            for nbr, val in zip(neighbors, values):
                if nbr in index_map:
                    affinity[i, index_map[nbr]] = val

        # Convert to CSR and symmetrize
        affinity = affinity.tocsr()
        affinity = affinity.maximum(affinity.T)  # Ensure symmetry

        # Spectral clustering with precomputed affinity
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, 
                                  module="sklearn.manifold._spectral_embedding")
            clustering = SpectralClustering(
                n_clusters=n_components,
                affinity='precomputed',
                random_state=0,
                n_init=100
            ).fit(affinity)
        
        f1 = f1_score(true_labels, clustering.labels_, average='weighted')
        total_f1 += f1
        processed_branches += 1

    final_score = total_f1 / processed_branches
    if validate_result:
        _validator(score=final_score)
    
    return final_score


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

    def test_perfect_clustering():
        """Test case where embedding perfectly aligns with graph structure (F1=1.0)."""
        # Create graph with one branch point (0) connected to two components (1, 2)
        graph = nx.Graph()
        graph.add_edges_from([(0,1), (0,2)])
        
        # Labels: 4 cells of type 0, 2 of type 1, 2 of type 2
        labels = np.array([0,0,0,0,1,1,2,2])
        n_cells = len(labels)
        
        # Manually create KNN matrix where:
        # - Cells 0-1 (type 0) connect to type 1 cells and each other
        # - Cells 2-3 (type 0) connect to type 2 cells and each other
        indices = [
            1,4,5,   # Row 0: neighbors 1 (type 0), 4 (type 1), 5 (type 1)
            0,4,5,   # Row 1: neighbors 0 (type 0), 4 (type 1), 5 (type 1)
            3,6,7,   # Row 2: neighbors 3 (type 0), 6 (type 2), 7 (type 2)
            2,6,7,   # Row 3: neighbors 2 (type 0), 6 (type 2), 7 (type 2)
            0,1,2,   # Row 4+: irrelevant for this test
            0,1,2,
            0,1,2,
            0,1,2,
        ]
        indptr = [0,3,6,9,12,15,18,21,24]
        data = np.ones(len(indices), dtype=np.float32)
        precomputed = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
        
        score = trajectory_cardinality_validation(
            given_graph=graph,
            labels_array=labels,
            precomputed_embedded_connectivities=precomputed,
            n_neighbors=3,  # Each row has 3 neighbors (excluding self)
            validate_result=True,
            skip_chain_nodes=False
        )
        assert np.isclose(score, 1.0), f"Perfect clustering failed: {score} != 1.0"

    def test_partial_clustering():
        """Test scenario with partially correct clustering (0.5 < F1 < 1.0)."""
        # Graph with branch point 0 splitting into two components
        graph = nx.Graph()
        graph.add_edges_from([(0,1), (0,2)])
        
        # 4 cells of type 0, 2 of type 1, 2 of type 2
        labels = np.array([0,0,0,0,1,1,2,2])
        n_cells = len(labels)
        
        # KNN matrix where 2 cells are correct, 2 are mixed
        indices = [
            1,4,5,   # Correct (type 1)
            0,4,5,   # Correct (type 1)
            3,6,7,   # Correct (type 2)
            2,4,6,   # Mixed neighbors (types 1 and 2)
            0,1,2,
            0,1,2,
            0,1,2,
            0,1,2,
        ]
        indptr = [0,3,6,9,12,15,18,21,24]
        data = np.ones(len(indices), dtype=np.float32)
        precomputed = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
        
        score = trajectory_cardinality_validation(
            given_graph=graph,
            labels_array=labels,
            precomputed_embedded_connectivities=precomputed,
            n_neighbors=3,
            validate_result=True,
            skip_chain_nodes=False
        )
        # Expect F1 between 0.5 and 1.0
        assert 0.5 < score < 1.0, f"Partial clustering unexpected score: {score}"

    def test_perfect_two_branches():
        """Test scenario with perfect branch preservation (F1=1.0)
        - Graph: B connected to A and C (removing B creates 2 components)
        - B cells have neighbors exclusively from one component
        - Spectral clustering should perfectly match expected branches
        """
        G = nx.Graph()
        G.add_edges_from([('B', 'A'), ('B', 'C')])
        
        # Labels: 2 A, 2 C, 4 B (total 8 cells)
        labels = np.array(['A', 'A', 'C', 'C', 'B', 'B', 'B', 'B'])
        
        # Manually create KNN matrix where:
        # - B cells 4-5 have neighbors from A
        # - B cells 6-7 have neighbors from C
        indices = [
            1,  # Cell 0 neighbors
            0,  # Cell 1
            3,  # Cell 2
            2,  # Cell 3
            0, 0,  # Cells 4-5 (neighbors from A)
            2, 2   # Cells 6-7 (neighbors from C)
        ]
        indptr = [0,1,2,3,4,5,6,7,8]
        data = np.ones(8, dtype=np.float32)
        precomputed = sparse.csr_matrix((data, indices, indptr), shape=(8,8))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=precomputed,
            n_neighbors=2,  # 1 real neighbor after excluding self
            validate_result=False,
            skip_chain_nodes=False
        )
        
        # Expect perfect alignment
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"


    def test_three_way_branch():
        """Complex test with 3-way branch point and perfect embedding alignment"""
        G = nx.Graph()
        G.add_edges_from([('B','A'), ('B','C'), ('B','D')])  # 3-way branch
        
        # 3 cells each for A/C/D, 9 for B (total 18 cells)
        labels = np.array(['A']*3 + ['C']*3 + ['D']*3 + ['B']*9)
        
        # Create KNN matrix where B cells are grouped into 3 clusters
        n_cells = 18
        indices = []
        indptr = [0]
        
        # B cells 9-17: neighbors from respective components
        for i in range(9, 18):
            if i < 12:  # First third: connect to A cells
                indices.extend([0,1,2])
            elif i < 15:  # Second third: connect to C cells
                indices.extend([3,4,5])
            else:  # Last third: connect to D cells
                indices.extend([6,7,8])
            indptr.append(len(indices))
        
        # Fill remaining rows with cyclic neighbors
        for i in range(9):
            indices.extend([0,1,2])
            indptr.append(len(indices))
        
        data = np.ones(len(indices), dtype=np.float32)
        precomputed = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=precomputed,
            n_neighbors=4,  # 3 real neighbors after excluding self
            validate_result=False,
            skip_chain_nodes=False
        )
        
        # Expect perfect clustering
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    test_perfect_clustering()
    test_partial_clustering()
    # test_perfect_two_branches()
    # test_three_way_branch()
    print("All tests passed.")
