#!/usr/bin/env python3

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score
from scipy import sparse
import warnings
from typing import Optional, Dict, Set, List
from loguru import logger
from collections import defaultdict

try:
    from sctram.evaluate._metrics._src.validators import validate_inclusive_between_0_1 as _validator
    from sctram.evaluate._metrics._src.utils import convert_scanpy_neighbors_to_indices
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator
    from utils import convert_scanpy_neighbors_to_indices


_logger = logger.bind(name="TrajectoryCardinalityValidator")


def trajectory_cardinality_validation(
    given_graph: nx.DiGraph,
    labels_array: np.ndarray,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    k_neighbors: int,
    validate_result: bool = True,
    skip_single_branches: bool = True,
    spectral_n_init: int = 20,
    min_cells_per_branch: int = 10,
    random_state: Optional[int] = None,
    exclude_self_from_neighbors: bool = True
) -> float:
    """Advanced Directed Trajectory Validation using Spectral Consensus (D-TVS)."""
    
    # Convert KNN connectivities to neighbor indices matrix
    embedded_neighbors = convert_scanpy_neighbors_to_indices(
        scanpy_neighbors_matrix=precomputed_embedded_connectivities,
        k=k_neighbors - 1,  # Critical correction: Scanpy stores k_neighbors-1 neighbors
        include_itself=not exclude_self_from_neighbors
    )

    # Identify branch points (nodes with out-degree ≥2)
    branch_points = [node for node in given_graph.nodes 
                     if given_graph.out_degree(node) >= 2]
    if not branch_points:
        _logger.warning("No biological branch points found")
        return np.nan
    
    total_f1 = 0.0
    valid_branches = 0
    label_to_nodes = defaultdict(set)
    for idx, lbl in enumerate(labels_array):
        label_to_nodes[lbl].add(idx)

    for bp in branch_points:
        _logger.debug(f"Processing branch point: {bp}")

        # Get biological branches via DAG reachability
        children = list(given_graph.successors(bp))
        if len(children) < 2:
            continue

        # Collect unique branches using directed descendants
        branch_sets = []
        for child in children:
            descendants = set()
            stack = [child]
            visited = set()
            while stack:
                node = stack.pop()
                if node in visited or node == bp:
                    continue
                visited.add(node)
                descendants.add(node)
                # Follow successors not leading back to branch point
                for succ in given_graph.successors(node):
                    if succ != bp and succ not in visited:
                        stack.append(succ)
            if descendants:
                branch_sets.append(descendants)

        # Remove duplicate branches using frozen sets
        unique_branches = []
        seen = set()
        for branch in branch_sets:
            frozen = frozenset(branch)
            if frozen not in seen:
                seen.add(frozen)
                unique_branches.append(branch)
        n_branches = len(unique_branches)
        if n_branches < 2:
            if skip_single_branches:
                _logger.debug(f"Skipping {bp}: only {n_branches} unique branches")
                continue
            else:
                raise ValueError(f"Branch {bp} has {n_branches} unique branches")

        # Create branch label mapping
        branch_mapper: Dict[str, int] = {}
        for bid, branch in enumerate(unique_branches):
            for node in branch:
                if node in branch_mapper:
                    _logger.warning(f"Node {node} appears in multiple branches")
                branch_mapper[node] = bid

        # Get cells of branch point type
        bp_cell_indices = label_to_nodes.get(bp, set())
        if not bp_cell_indices:
            _logger.warning(f"No cells found for branch point {bp}")
            continue
        bp_cells = np.array(sorted(bp_cell_indices))
        if len(bp_cells) < min_cells_per_branch * n_branches:
            _logger.warning(f"Skipping {bp}: insufficient cells")
            continue

        # Neighbor voting to determine expected clusters
        expected_labels = []
        valid_indices = []
        for cell_idx in bp_cells:
            # Correct neighbor index extraction from CSR
            start = embedded_neighbors.indptr[cell_idx]
            end = embedded_neighbors.indptr[cell_idx + 1]
            neighbors = embedded_neighbors.indices[start:end]
            
            if exclude_self_from_neighbors:
                # Filter out self-reference if present
                neighbors = [nbr for nbr in neighbors if nbr != cell_idx]

            votes = []
            for nbr_idx in neighbors:
                nbr_label = labels_array[nbr_idx]
                if nbr_label == bp:
                    continue  # Exclude same-type neighbors
                if nbr_label in branch_mapper:
                    votes.append(branch_mapper[nbr_label])
            
            if not votes:
                continue  # Skip cells with no informative neighbors
            
            # Deterministic majority voting
            unique, counts = np.unique(votes, return_counts=True)
            max_count = counts.max()
            candidates = unique[counts == max_count]
            chosen = min(candidates)  # Consistent tie-break
            expected_labels.append(chosen)
            valid_indices.append(cell_idx)

        # Validate remaining cells
        valid_indices = np.array(valid_indices)
        if len(valid_indices) < min_cells_per_branch * n_branches:
            _logger.warning(f"Skipping {bp}: insufficient valid cells")
            continue
        unique_expected = np.unique(expected_labels)
        if len(unique_expected) < n_branches:
            _logger.warning(f"Skipping {bp}: incomplete branch coverage")
            continue

        # Create spectral clustering input
        cell_mask = np.isin(bp_cells, valid_indices)
        subset_cells = bp_cells[cell_mask]
        affinity_subset = precomputed_embedded_connectivities[subset_cells][:, subset_cells]

        # Perform spectral clustering
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                clustering = SpectralClustering(
                    n_clusters=n_branches,
                    affinity='precomputed',
                    n_init=spectral_n_init,
                    random_state=random_state
                ).fit(affinity_subset)
            cluster_labels = clustering.labels_
        except Exception as e:
            _logger.error(f"Spectral clustering failed for {bp}: {str(e)}")
            continue

        # Calculate validation metric
        f1 = f1_score(expected_labels, cluster_labels, average='weighted')
        total_f1 += f1
        valid_branches += 1
        _logger.debug(f"Branch {bp} F1: {f1:.3f}")

    if valid_branches == 0:
        _logger.warning("No valid branches processed")
        return np.nan
    
    final_score = total_f1 / valid_branches
    if validate_result:
        _validator(score=final_score)
    
    return final_score


def test_perfect_alignment():
    """Perfect alignment between graph branches and spectral clusters (F1=1.0)."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E')])
    
    # Create synthetic data with clear branch separation
    labels = np.array(['A']*100 + ['B']*50 + ['C']*50 + ['D']*30 + ['E']*30)
    n_cells = len(labels)
    
    # Create ideal connectivity matrix (A cells connect to B/C clusters)
    row = np.repeat(np.arange(100), 10)
    col = np.concatenate([np.random.choice(100 + np.arange(100), 10, replace=False) 
                         for _ in range(100)])
    data = np.ones_like(row)
    conn = sparse.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
    
    score = trajectory_cardinality_validation(
        G, labels, conn, k_neighbors=10, spectral_n_init=1, random_state=42
    )
    assert np.isclose(score, 1.0), f"Perfect alignment failed: {score}"

def test_single_branch_skipped():
    """Branch point with only one valid branch should be skipped."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C')])  # No true bifurcation
    
    labels = np.array(['A', 'B', 'C'])
    conn = sparse.csr_matrix(np.eye(3))
    
    score = trajectory_cardinality_validation(G, labels, conn, k_neighbors=1)
    assert np.isnan(score), f"Single branch handling failed: {score}"

def test_complete_mismatch():
    """Complete mismatch between graph and embedding (F1≈0)."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C')])
    
    # Create deceptive embedding where all A neighbors are type B
    labels = np.array(['A']*100 + ['B']*100 + ['C']*100)
    row = np.repeat(np.arange(100), 10)
    col = np.random.choice(100 + np.arange(200), 1000, replace=False)
    data = np.ones(1000)
    conn = sparse.csr_matrix((data, (row, col)), shape=(300, 300))
    
    score = trajectory_cardinality_validation(
        G, labels, conn, k_neighbors=10, spectral_n_init=1, random_state=42
    )
    assert score < 0.3, f"Mismatch detection failed: {score}"

def test_large_manual_validation():
    """Large-scale validation with predictable structure (manual F1 calculation)."""
    np.random.seed(42)
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C')])
    
    # 10,000 cells with perfect branch separation
    labels = np.array(['A']*5000 + ['B']*2500 + ['C']*2500)
    n_cells = len(labels)
    
    # Create ideal bipartite connectivity
    row = np.concatenate([
        np.repeat(np.arange(5000), 2500),  # A->B connections
        np.repeat(np.arange(5000), 2500)   # A->C connections
    ])
    col = np.concatenate([
        np.random.choice(5000 + np.arange(2500), 5000*2500),
        np.random.choice(7500 + np.arange(2500), 5000*2500)
    ])
    data = np.ones_like(row)
    conn = sparse.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
    
    score = trajectory_cardinality_validation(
        G, labels, conn, k_neighbors=50, spectral_n_init=1, random_state=42
    )
    assert np.isclose(score, 1.0, atol=0.01), f"Large test failed: {score}"

def test_no_branch_points():
    """Graph without branch points should return NaN."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C')])
    
    labels = np.array(['A', 'B', 'C'])
    conn = sparse.eye(3)
    
    score = trajectory_cardinality_validation(G, labels, conn, k_neighbors=1)
    assert np.isnan(score), f"No branch point handling failed: {score}"

def test_insufficient_cells():
    """Branch points with insufficient cells should be skipped."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C')])
    
    # Only 5 cells at branch point A (needs at least 10 per branch)
    labels = np.array(['A']*5 + ['B']*20 + ['C']*20)
    conn = sparse.csr_matrix([1]*len(labels), (np.arange(len(labels)), 
                            np.arange(len(labels))), shape=(len(labels), len(labels)))
    
    score = trajectory_cardinality_validation(
        G, labels, conn, k_neighbors=1, min_cells_per_branch=10
    )
    assert np.isnan(score), f"Insufficient cells handling failed: {score}"

def test_deterministic_tie_breaking():
    """Tie-breaking in neighbor voting should be deterministic."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C')])
    
    # Create balanced votes between branches 0 and 1
    labels = np.array(['A']*1 + ['B']*50 + ['C']*50)
    row = [0]*10  # Single A cell
    col = list(range(1, 51))[:5] + list(range(51, 101))[:5]  # 5 B + 5 C neighbors
    data = [1]*10
    conn = sparse.csr_matrix((data, (row, col)), shape=(101, 101))
    
    score = trajectory_cardinality_validation(
        G, labels, conn, k_neighbors=10, spectral_n_init=1, random_state=42
    )
    
    # Expected to choose branch 0 (B) due to deterministic tie-break
    assert np.isclose(score, 1.0), f"Tie-breaking failed: {score}"

def create_realistic_connectivity(labels, k, noise=0.1):
    """Create realistic KNN matrix with controlled noise."""
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(42)
    
    # Create embedding coordinates with cluster structure
    cluster_centers = {'A': [0,0], 'B': [5,5], 'C': [5,-5]}
    coords = np.vstack([cluster_centers[lbl] + np.random.randn(len(labels)//3, 2)*0.5 
                      for lbl in np.unique(labels)])
    
    # Add Gaussian noise
    coords += np.random.normal(0, noise, coords.shape)
    
    # Compute KNN
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    return nn.kneighbors_graph(mode='connectivity')

def test_real_world_simulation():
    """Real-world simulation with ground truth F1 calculation."""
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C')])
    
    # Generate 3000 cells with clear branch structure
    labels = np.array(['A']*1000 + ['B']*1000 + ['C']*1000)
    conn = create_realistic_connectivity(labels, k=15, noise=0.3)
    
    # Manually calculate expected F1
    bp_cells = np.where(labels == 'A')[0]
    expected_labels = np.zeros(len(bp_cells))
    cluster_labels = SpectralClustering(n_clusters=2, random_state=42).fit(conn[bp_cells][:, bp_cells]).labels_
    
    manual_f1 = f1_score(expected_labels, cluster_labels, average='weighted')
    
    # Compute metric
    metric_f1 = trajectory_cardinality_validation(
        G, labels, conn, k_neighbors=15, spectral_n_init=10, random_state=42
    )
    
    assert np.isclose(metric_f1, manual_f1, rtol=0.05), \
        f"Real-world test failed: {metric_f1} vs {manual_f1}"

if __name__ == '__main__':
    test_perfect_alignment()
    test_single_branch_skipped()
    test_complete_mismatch()
    test_large_manual_validation()
    test_no_branch_points()
    test_insufficient_cells()
    test_deterministic_tie_breaking()
    test_real_world_simulation()
    print("All trajectory validation tests passed!")