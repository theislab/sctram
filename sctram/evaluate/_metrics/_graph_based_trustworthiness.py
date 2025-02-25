#!/usr/bin/env python3

import numpy as np
import networkx as nx
from bisect import bisect_left, bisect_right
from scipy import sparse
from collections import defaultdict

try:
    from sctram.evaluate._metrics.validators import validate_between_minus_plus_1 as _validator
    from sctram.evaluate._metrics.utils import convert_scanpy_neighbors_to_indices
except ImportError:
    from validators import validate_between_minus_plus_1 as _validator
    from utils import convert_scanpy_neighbors_to_indices


def graph_based_trustworthiness(
    given_graph: nx.Graph,
    labels_array: np.ndarray,
    n_neighbors: int,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    validate_result: bool = True
) -> float:
    """Computes graph-based trustworthiness between cell type graph and embedding.

    Statistical Foundation:
        - Trustworthiness measures preservation of local neighborhoods between input (graph) 
          and output (embedding) spaces. Graph distances between cell types define ground truth 
          biological relationships. Embedding neighborhoods should reflect these relationships.

    Mathematical Formulation:
        T(k) = 1 - (Σ penalties) / (Σ max_penalties)
        Penalty for cross-type neighbor j of cell i: max(rank(j|i) - allowed_cross(i), 0)
        allowed_cross(i) = max(k - same_type_count(i), 0)
        max_penalties per cell: min(k, cross_type_count(i)) * max(cross_type_count(i) - allowed_cross(i), 0)

    Statistical Justification:
        - Penalizes cross-type neighbors in embedding that violate graph hierarchy.
        - Normalization accounts for varying cell type distributions, ensuring scores are 
          comparable across datasets.

    Advantages:
        - Handles arbitrary cell type hierarchies via graph input.
        - Efficient rank calculation using precomputed graph distances and binary search.
        - Robust normalization adapts to varying cell type frequencies.

    Limitations:
        - Assumes cell type graph fully captures biological relationships.
        - Shortest paths may not capture all biological nuances.
        - Preprocessing graph distances is O(t^2) for t cell types.

    Parameters:
        given_graph (nx.Graph): Cell type relationship graph with nodes as cell types.
        labels_array (np.ndarray): Cell type labels for each cell (length n).
        n_neighbors (int): Number of neighbors considered (k).
        precomputed_embedded_connectivities (sparse.csr_matrix): Precomputed k-nearest neighbor connectivities.
        validate_result (bool): Whether to validate the score is between 0 and 1.

    Returns:
        float: Trustworthiness score between 0 (worst) and 1 (best).
    """
    n_samples = labels_array.shape[0]
    if n_neighbors >= n_samples // 2 and n_neighbors>3:
        raise ValueError(f"n_neighbors must be < {n_samples//2}")
    
    embedded_neighbors = convert_scanpy_neighbors_to_indices(
        scanpy_neighbors_matrix = precomputed_embedded_connectivities,
        k = n_neighbors - 1,  # excluding itself
        include_itself = False
    )  # this should create neighbor indices except itself.

    # Precompute graph relationships and type mappings
    cell_type_info, type_distance = _preprocess_graph(given_graph, labels_array)

    # Compute trustworthiness score
    trust = _compute_trustworthiness(
        labels_array,
        embedded_neighbors,
        cell_type_info,
        type_distance,
        n_neighbors - 1,
        n_samples
    )

    if validate_result:
        _validator(score=trust)

    return trust

def _preprocess_graph(graph: nx.Graph, labels: np.ndarray):
    """Precompute graph relationships and cell type mappings."""
    cell_types = list(graph.nodes())
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    # Create cell type to indices mapping
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_to_indices[lbl].append(idx)
    for lbl in label_to_indices:
        label_to_indices[lbl].sort()  # Ensure sorted for binary search

    # Precompute cell type distance relationships (excluding self type)
    cell_type_info = {}
    for src in cell_types:
        # Exclude src type from targets and sort by (distance, type)
        targets = sorted(
            [(dst, shortest_paths[src].get(dst, float('inf'))) for dst in cell_types if dst != src],
            key=lambda x: (x[1], x[0]))
        cumulative_counts = []
        total = 0
        for dst, dist in targets:
            count = len(label_to_indices.get(dst, []))
            total += count
            cumulative_counts.append((dist, total))  # (distance, cumulative_cell_count)
        
        cell_type_info[src] = {
            'sorted_targets': targets,
            'cumulative_counts': cumulative_counts,
            'indices_map': label_to_indices
        }

    # Precompute type-to-type distances
    type_distance = defaultdict(lambda: defaultdict(int))
    for src in cell_types:
        for dst in cell_types:
            type_distance[src][dst] = shortest_paths[src].get(dst, float('inf'))

    return cell_type_info, type_distance


def _convert_scanpy_neighbors_to_indices(scanpy_neighbors_distances: sparse.csr_matrix) -> sparse.csr_matrix:
    """Converts Scanpy's neighbors distances matrix to indices matrix for trustworthiness calculation.
    
    Parameters:
        scanpy_neighbors_distances (sparse.csr_matrix): Distances matrix from 
            `sc.pp.neighbors()`, shape (n_cells, n_total_cells). Each row contains 
            distances to nearest neighbors sorted in ascending order.
    
    Returns:
        sparse.csr_matrix: CSR matrix where each row contains column indices from 
        input distances matrix, representing embedding neighbor indices sorted by distance.
    """
    n_cells = scanpy_neighbors_distances.shape[0]
    indices_list = []
    
    # Extract column indices for each row from distances CSR matrix
    for i in range(n_cells):
        start = scanpy_neighbors_distances.indptr[i]
        end = scanpy_neighbors_distances.indptr[i+1]
        indices_list.append(scanpy_neighbors_distances.indices[start:end])
    
    # Create dense array of neighbor indices and convert to CSR
    embedded_indices = np.vstack(indices_list)
    return sparse.csr_matrix(embedded_indices)


def _compute_trustworthiness(labels, embedded_neighbors, cell_type_info, type_distance, k, n_samples):
    """Core trustworthiness calculation with dynamic normalization."""
    penalty = 0.0
    label_counts = defaultdict(int)
    for lbl in labels:
        label_counts[lbl] += 1

    # Precompute max_penalty components for each cell
    total_max_penalty = 0.0
    max_penalty_components = []
    for i in range(n_samples):
        src_type = labels[i]
        m_i = label_counts[src_type] - 1  # Exclude self
        C_i = n_samples - m_i - 1  # Cross-type cells count
        t_i = max(k - m_i, 0)
        x_i = min(k, C_i)
        contribution = x_i * max(C_i - t_i, 0)
        max_penalty_components.append(contribution)
        total_max_penalty += contribution

    # Compute penalty by comparing embedded neighbors to graph structure
    for i in range(n_samples):
        src_type = labels[i]
        src_info = cell_type_info[src_type]
        sorted_targets = src_info['sorted_targets']
        cum_counts = src_info['cumulative_counts']
        indices_map = src_info['indices_map']

        m_i = label_counts[src_type] - 1
        t_i = max(k - m_i, 0)
        if t_i <= 0:
            # All cross-type neighbors are penalized
            allowed_cross = 0
        else:
            allowed_cross = t_i

        for j in embedded_neighbors[i]:
            if j == i:
                continue  # Skip self
            
            j_type = labels[j]
            if j_type == src_type:
                continue  # Same type, no penalty
            
            # Calculate cross-type rank
            distance = type_distance[src_type][j_type]
            targets = [t for t, _ in sorted_targets]
            distances = [d for _, d in sorted_targets]

            # Find position in sorted targets
            idx = bisect_left(distances, distance)
            start_idx = bisect_left(distances, distance)
            end_idx = bisect_right(distances, distance)

            # Find number of types with distance < current or same distance and type < j_type
            same_dist_targets = sorted_targets[start_idx:end_idx]
            types_before = 0
            for t, _ in same_dist_targets:
                if t < j_type:
                    types_before += len(indices_map.get(t, []))
                elif t == j_type:
                    break

            base_rank = cum_counts[start_idx-1][1] if start_idx > 0 else 0
            base_rank += types_before

            # Position within j_type's cells
            type_indices = indices_map.get(j_type, [])
            pos = bisect_left(type_indices, j)
            rank = base_rank + pos + 1  # 1-based

            # Apply penalty if rank exceeds allowed_cross
            excess = rank - allowed_cross
            if excess > 0:
                penalty += excess

    # Calculate final score
    if total_max_penalty > 0:
        trust = 1.0 - (penalty / total_max_penalty)
    else:
        trust = 1.0  # All neighbors are perfectly preserved
    
    return trust

if __name__ == "__main__":
    
    def test_perfect_alignment():
        """Test when embedding perfectly aligns with graph structure (score=1.0)."""
        given_graph = nx.Graph()
        given_graph.add_edges_from([('A', 'B'), ('B', 'C')])
        labels = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
        
        # All neighbors are same-type (no cross-type)
        indices = [
            [1, 2], [0, 2], [0, 1],  # A (0,1,2)
            [4, 5], [3, 5], [3, 4],  # B (3,4,5)
            [7, 8], [6, 8], [6, 7]   # C (6,7,8)
        ]
        embedded_conn = _create_csr_matrix(indices, (9,9))
        score = graph_based_trustworthiness(given_graph, labels, 3, embedded_conn)
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def test_manual_example():
        """Test manually calculated example with known score (5/6 ≈0.8333)."""
        given_graph = nx.Graph([('A', 'B'), ('B', 'C')])
        labels = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        indices = [
            [2, 3], [2, 3],  # A0, A1
            [0, 1], [0, 1],  # B0, B1
            [2, 3], [2, 3]   # C0, C1
        ]
        embedded_conn = _create_csr_matrix(indices, (6,6))
        score = graph_based_trustworthiness(given_graph, labels, 3, embedded_conn)
        expected = 5/6
        assert np.isclose(score, expected, rtol=1e-5), f"Expected {expected}, got {score}"

    def test_disconnected_graph():
        """Test disconnected graph with cross-type neighbors (score=0.5)."""
        given_graph = nx.Graph()
        given_graph.add_nodes_from(['A', 'B'])  # No edge between A and B
        labels = np.array(['A', 'A', 'B', 'B'])
        indices = [[2], [2], [0], [0]]  # Each cell has 1 cross-type neighbor
        embedded_conn = _create_csr_matrix(indices, (4,4))
        score = graph_based_trustworthiness(given_graph, labels, 2, embedded_conn)
        assert np.isclose(score, 0.5), f"Expected 0.5, got {score}"

    def test_single_cell_type():
        """Test single cell type (no cross-type possible, score=1.0)."""
        given_graph = nx.Graph()
        given_graph.add_node('A')
        labels = np.array(['A'] * 5)
        indices = [[1,2], [0,2], [0,1], [0,1], [1,2]]  # Arbitrary same-type
        embedded_conn = _create_csr_matrix(indices, (5,5))
        score = graph_based_trustworthiness(given_graph, labels, 3, embedded_conn)
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def test_large_hierarchy():
        """Test large hierarchy with perfect embedding (score=1.0)."""
        # Create a tree graph: A -> B -> C -> D
        given_graph = nx.Graph([('A','B'), ('B','C'), ('C','D')])
        labels = np.array(['A']*5 + ['B']*5 + ['C']*5 + ['D']*5)
        # Neighbors follow hierarchy: same-type, then next closest type
        indices = []
        for i in range(20):
            if i <5:  # A: neighbors are A (0-4)
                neighbors = [j for j in range(5) if j !=i][:4]
            elif i<10: # B: neighbors B and A
                neighbors = [j for j in range(5,10) if j !=i]
                neighbors += [0,1,2,3][:4-len(neighbors)]
            elif i<15: # C: neighbors C and B
                neighbors = [j for j in range(10,15) if j !=i]
                neighbors += [5,6,7,8][:4-len(neighbors)]
            else: # D: neighbors D and C
                neighbors = [j for j in range(15,20) if j !=i]
                neighbors += [10,11,12,13][:4-len(neighbors)]
            indices.append(neighbors[:4])  # Use 4 neighbors (k=4)
        embedded_conn = _create_csr_matrix(indices, (20,20))
        score = graph_based_trustworthiness(given_graph, labels, 5, embedded_conn)
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def _create_csr_matrix(indices, shape):
        """Helper to create a CSR matrix from neighbor indices."""
        indptr = [0]
        data = []
        csr_indices = []
        for row in indices:
            csr_indices.extend(row)
            data.extend([1]*len(row))
            indptr.append(len(csr_indices))
        return sparse.csr_matrix((data, csr_indices, indptr), shape=shape)

    
    test_perfect_alignment()
    test_manual_example()
    test_disconnected_graph()
    test_single_cell_type()
    test_large_hierarchy()
    print("All tests passed.")