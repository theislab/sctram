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
except ImportError:
    from validators import validate_inclusive_between_0_1 as _validator

_logger = logger.bind(name="TrajectoryValidator")


def trajectory_cardinality_validation(
    given_graph: nx.DiGraph,
    labels_array: np.ndarray,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    validate_result: bool = True,
    skip_single_branches: bool = True,
    spectral_n_init: int = 50,
    min_cells_per_branch: int = 15,
    random_state: Optional[int] = None,
) -> float:
    """Computes the Directed Trajectory Validation Score (DTVS) between a cell embedding and a trajectory graph.
    
    DTVS (Directed Trajectory Validation Score) is a robust metric designed to quantitatively evaluate how 
    well a cell embedding (e.g., from single-cell genomics data) aligns with a biological trajectory graph. 
    It focuses on validating whether the local structure of the embedding reflects the branching patterns defined
    in the trajectory graph, which represents hypothesized developmental pathways.
    
    This advanced metric evaluates topological consistency by comparing spectral clustering results in the embedding
    with biological branches derived from neighbor voting. Designed for single-cell genomics data, it handles:
    - Complex branching trajectories
    - High-dimensional embeddings
    - Sparse biological graphs
    
    Statistical Rationale:
    1. Local Consistency: Assumes cells near branch points should have neighbors predictive of downstream branches
    2. Manifold Alignment: Spectral clustering captures non-linear structures in the embedding
    3. Weighted Scoring: Uses F1-score to handle class imbalance between branches
    
    Args:
        given_graph: Biological trajectory as directed acyclic graph (DAG)
        labels_array: Cell-type labels for each cell
        precomputed_embedded_connectivities: Scanpy neighbors connectivities matrix
        validate_result: Ensure output is in [0,1]
        skip_single_branches: Skip branch points with <2 branches
        spectral_n_init: Spectral clustering initializations for stability
        min_cells_per_branch: Minimum cells required per branch
        random_state: Random seed for reproducibility
    
    Returns:
        float: DTVS score between 0 (no alignment) and 1 (perfect alignment)
    
    Raises:
        ValueError: On invalid input dimensions or empty graph
    """
    # Input validation
    n_cells = labels_array.shape[0]
    if precomputed_embedded_connectivities.shape != (n_cells, n_cells):
        raise ValueError("Connectivities matrix must be n_cells x n_cells")
    
    if not given_graph.nodes:
        raise ValueError("Input graph contains no nodes")
    
    # Precompute label indices
    label_indices = defaultdict(set)
    for idx, lbl in enumerate(labels_array):
        label_indices[lbl].add(idx)
    
    # Identify valid branch points
    branch_points = [node for node in given_graph.nodes 
                    if given_graph.out_degree(node) >= 2]
    if not branch_points:
        raise ValueError("No biological branch points found")
    
    total_score = 0.0
    processed_branches = 0
    
    for bp in branch_points:
        # Get all immediate children
        children = list(given_graph.successors(bp))
        if len(children) < 2:
            continue
        
        # Extract unique biological branches
        branch_sets = []
        visited_global = set()
        for child in children:
            if child in visited_global:
                continue
            # Collect all descendants not revisiting branch point
            descendants = set(nx.descendants(given_graph, child))
            descendants.add(child)
            # Exclude nodes from other branches and branch point
            clean_desc = descendants - {bp}
            if not clean_desc:
                continue
            branch_sets.append(clean_desc)
            visited_global.update(clean_desc)
        
        # Remove duplicate branches using containment checks
        unique_branches = []
        for branch in branch_sets:
            is_unique = True
            for existing in unique_branches:
                if branch.issubset(existing) or existing.issubset(branch):
                    is_unique = False
                    break
            if is_unique:
                unique_branches.append(branch)
        
        n_branches = len(unique_branches)
        if n_branches < 2:
            if skip_single_branches:
                _logger.debug(f"Skipping {bp}: {n_branches} unique branches")
                continue
            else:
                raise ValueError(f"Branch point {bp} has {n_branches} branches")
        
        # Create branch membership mapping
        branch_map = {}
        for bid, nodes in enumerate(unique_branches):
            for node in nodes:
                if node in branch_map:
                    _logger.warning(f"Node {node} in multiple branches")
                branch_map[node] = bid
        
        # Get cells for current branch point
        bp_cells = np.array(sorted(label_indices.get(bp, [])))
        if len(bp_cells) < min_cells_per_branch * n_branches:
            _logger.debug(f"Skipping {bp}: insufficient cells")
            continue
        
        # Neighbor voting process
        expected_labels = []
        valid_cells = []
        for cell_idx in bp_cells:
            # Extract neighbors from precomputed CSR
            start = precomputed_embedded_connectivities.indptr[cell_idx]
            end = precomputed_embedded_connectivities.indptr[cell_idx + 1]
            neighbors = precomputed_embedded_connectivities.indices[start:end]
            
            # Collect votes from neighbors in downstream branches
            votes = []
            for nbr in neighbors:
                nbr_label = labels_array[nbr]
                if nbr_label == bp:
                    continue  # Ignore same-type neighbors
                if nbr_label in branch_map:
                    votes.append(branch_map[nbr_label])
            
            if not votes:
                continue  # No informative neighbors
            
            # Stable voting with sorted tie-break
            unique, counts = np.unique(votes, return_counts=True)
            max_votes = counts.max()
            candidates = unique[counts == max_votes]
            chosen_branch = sorted(candidates)[0]  # Deterministic tie-break
            
            expected_labels.append(chosen_branch)
            valid_cells.append(cell_idx)
        
        # Check minimum cell requirements
        valid_cells = np.array(valid_cells)
        if len(valid_cells) < min_cells_per_branch * n_branches:
            _logger.debug(f"Skipping {bp}: insufficient valid cells")
            continue
        if len(np.unique(expected_labels)) < n_branches:
            _logger.debug(f"Skipping {bp}: incomplete branch coverage")
            continue
        
        # Spectral clustering on embedding connectivities
        try:
            # Subset connectivities matrix
            cell_mask = np.isin(bp_cells, valid_cells)
            subset_indices = bp_cells[cell_mask]
            affinity = precomputed_embedded_connectivities[subset_indices, :][:, subset_indices]
            
            # Ensure connectivity
            if affinity.nnz == 0:
                raise ValueError("Affinity matrix is completely disconnected")
            
            # Perform clustering
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                clusterer = SpectralClustering(
                    n_clusters=n_branches,
                    affinity='precomputed',
                    n_init=spectral_n_init,
                    random_state=random_state
                )
                cluster_labels = clusterer.fit_predict(affinity.toarray())
            
            # Calculate alignment score
            f1 = f1_score(expected_labels, cluster_labels, average='weighted')
            total_score += f1
            processed_branches += 1
            _logger.debug(f"Branch {bp} score: {f1:.3f}")
        
        except Exception as e:
            _logger.error(f"Validation failed for {bp}: {str(e)}")
            continue
    
    if processed_branches == 0:
        raise ValueError("No valid branches processed")
    
    final_score = total_score / processed_branches
    if validate_result:
        _validator(final_score)
    
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
            row_neighbors = [(i + j) % n_cells for j in range(k)]
            indices.extend(row_neighbors)
            indptr.append(len(indices))
        
        data = np.ones(len(indices), dtype=np.float32)
        return sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))

    def test_perfect_alignment():
        """Test scenario with perfect alignment between graph structure and embedding"""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C')])
        labels = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'C', 'C'])
        
        # Manually construct ideal connectivities matrix
        indices = [
            1,4,  # Row 0
            0,4,  # Row 1
            3,6,  # Row 2
            2,6,  # Row 3
            5,6,  # Row 4 (B cells)
            4,7,  # Row 5 (B cells)
            7,0,  # Row 6 (C cells)
            6,1   # Row 7 (C cells)
        ]
        indptr = [0,2,4,6,8,10,12,14,16]
        data = np.ones(len(indices), dtype=np.float32)
        connectivities = sparse.csr_matrix((data, indices, indptr), shape=(8, 8))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            validate_result=False,
            skip_single_branches=False,
            spectral_n_init=10,
            min_cells_per_branch=2,
            random_state=42
        )
        assert np.isclose(score, 1.0), f"Perfect alignment failed. Expected 1.0, got {score:.4f}"

    def test_no_branch_points():
        """Test scenario with no branch points in graph"""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B')])
        labels = np.array(['A', 'B'])
        connectivities = create_valid_scanpy_neighbors(n_cells=2, k=1)
        
        try:
            score = trajectory_cardinality_validation(
                given_graph=G,
                labels_array=labels,
                precomputed_embedded_connectivities=connectivities,
            )
            assert False, f"No branch points failed. Expected error raised, got {score}"
        except ValueError:
            pass
        
    def test_insufficient_cells():
        """Test scenario with insufficient cells per branch"""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C')])
        labels = np.array(['A']*3 + ['B']*2 + ['C']*2)
        connectivities = create_valid_scanpy_neighbors(n_cells=7, k=2)
        
        try:
            score = trajectory_cardinality_validation(
                given_graph=G,
                labels_array=labels,
                precomputed_embedded_connectivities=connectivities,
                min_cells_per_branch=2
            )
            assert False, f"Insufficient cells failed. Expected error raised, got {score}"
        except ValueError:
            pass

    def test_misaligned_clusters():
        """Test scenario where clustering results disagree with biological branches"""
        G = nx.DiGraph([('A', 'B'), ('A', 'C')])
        labels = np.array(['A']*4 + ['B']*2 + ['C']*2)
        
        # Connectivity pattern forcing spectral clustering disagreement
        indices = [
            # A cells (0-3) form cross-branch connections
            2, 4,  # A0 -> A2 (C branch), B4
            3, 5,  # A1 -> A3 (C branch), B5
            0, 6,  # A2 -> A0 (B branch), C6
            1, 7,  # A3 -> A1 (B branch), C7
            # B cells (4-5) connect to both A clusters
            0, 2,  # B4 -> A0, A2
            1, 3,  # B5 -> A1, A3
            # C cells (6-7) connect to both A clusters
            0, 2,  # C6 -> A0, A2
            1, 3   # C7 -> A1, A3
        ]
        indptr = [0,2,4,6,8,10,12,14,16]
        data = np.ones(len(indices), dtype=np.float32)
        connectivities = sparse.csr_matrix((data, indices, indptr), shape=(8,8))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            spectral_n_init=10,
            min_cells_per_branch=2,
            random_state=42,
            skip_single_branches=False
        )
        
        assert 0.4 <= score <= 0.6, f"Expected ~0.5, got {score:.4f}"
                
    def test_large_scale_validation():
        """Large-scale test with 1000 cells to verify computational efficiency"""
        np.random.seed(42)
        n_cells = 1000
        labels = np.array(['A']*400 + ['B']*300 + ['C']*200 + ['D']*100)
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D')])
        
        # Generate biologically structured connectivities
        def create_structured_connectivities():
            indices = []
            indptr = [0]
            k = 15
            
            # Label ranges
            label_ranges = {
                'A': (0, 400),
                'B': (400, 700),
                'C': (700, 900),
                'D': (900, 1000)
            }
            
            for i in range(n_cells):
                current_label = labels[i]
                start, end = label_ranges[current_label]
                
                # Base neighbors: 80% from current label, 20% from children
                if current_label == 'A':
                    pool = list(range(start, end)) + \
                        list(range(400, 700)) + \
                        list(range(700, 900))
                elif current_label == 'B':
                    pool = list(range(400, 700)) + \
                        list(range(900, 1000))
                elif current_label == 'C':
                    pool = list(range(700, 900))
                else:  # D
                    pool = list(range(900, 1000))
                
                # Randomly select k neighbors with biological structure
                neighbors = np.random.choice(pool, size=k, replace=True)
                indices.extend(neighbors)
                indptr.append(len(indices))
            
            data = np.ones(len(indices), dtype=np.float32)
            return sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
        
        connectivities = create_structured_connectivities()
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            min_cells_per_branch=50,
            spectral_n_init=10,
            random_state=42
        )
        
        # Validate score falls in reasonable range
        assert 0 <= score <= 1, f"Invalid score range: {score}"
        assert not np.isnan(score), "Large scale test failed with nan"
        
    def test_nested_branch_perfect_alignment():
        """Test nested branch points with perfect alignment"""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E')])
        labels = np.array(['A']*400 + ['B']*200 + ['C']*200 + ['D']*150 + ['E']*150)
        n_cells = len(labels)
        
        # Create structured connectivities with clear branch separation
        indices = []
        indptr = [0]
        k = 15
        label_ranges = {
            'A': (0, 400),
            'B': (400, 600),
            'C': (600, 800),
            'D': (800, 950),
            'E': (950, 1100)
        }
        
        # Split A cells into two groups connecting to B/C respectively
        for i in range(n_cells):
            lbl = labels[i]
            if lbl == 'A':
                if i < 200:  # First half connects to B
                    neighbors = np.random.choice(np.arange(*label_ranges['B']), k)
                else:        # Second half connects to C
                    neighbors = np.random.choice(np.arange(*label_ranges['C']), k)
            elif lbl == 'B':
                # Split B cells between D/E connections
                if i < 500:
                    neighbors = np.random.choice(np.arange(*label_ranges['D']), k)
                else:
                    neighbors = np.random.choice(np.arange(*label_ranges['E']), k)
            else:
                start, end = label_ranges[lbl]
                neighbors = np.random.choice(np.arange(start, end), k)
            
            indices.extend(neighbors)
            indptr.append(len(indices))
        
        data = np.ones(len(indices), dtype=np.float32)
        connectivities = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            min_cells_per_branch=50,
            spectral_n_init=10,
            random_state=42
        )
        assert np.isclose(score, 1.0, atol=0.05), f"Nested alignment failed. Expected ~1.0, got {score}"

    def test_large_scale_perfect_clusters():
        """Large-scale test with 10k cells and clear branch separation"""
        np.random.seed(42)
        n_cells = 10000
        labels = np.array(['A']*4000 + ['B']*3000 + ['C']*2000 + ['D']*1000)
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D')])
        
        indices = []
        indptr = [0]
        k = 15
        label_ranges = {
            'A': (0, 4000),
            'B': (4000, 7000),
            'C': (7000, 9000),
            'D': (9000, 10000)
        }
        
        # Create perfect branch connections
        for i in range(n_cells):
            lbl = labels[i]
            if lbl == 'A':
                # Split A cells between B/C connections
                if i < 2000:
                    neighbors = np.random.choice(label_ranges['B'], k)
                else:
                    neighbors = np.random.choice(label_ranges['C'], k)
            elif lbl == 'B':
                neighbors = np.random.choice(label_ranges['D'], k)
            else:
                start, end = label_ranges[lbl]
                neighbors = np.random.choice(range(start, end), k)
            
            indices.extend(neighbors)
            indptr.append(len(indices))
        
        data = np.ones(len(indices), dtype=np.float32)
        connectivities = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            min_cells_per_branch=500,
            spectral_n_init=10,
            random_state=42
        )
        assert np.isclose(score, 1.0, atol=0.05), f"Large test failed. Expected ~1.0, got {score}"

    def test_three_way_branching():
        """Test three-way branching with clear separation"""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('A', 'D')])
        labels = np.array(['A']*600 + ['B']*200 + ['C']*200 + ['D']*200)
        n_cells = len(labels)
        
        indices = []
        indptr = [0]
        k = 15
        label_ranges = {
            'A': (0, 600),
            'B': (600, 800),
            'C': (800, 1000),
            'D': (1000, 1200)
        }
        
        # Split A cells into three groups
        for i in range(n_cells):
            lbl = labels[i]
            if lbl == 'A':
                if i < 200:
                    neighbors = np.random.choice(label_ranges['B'], k)
                elif i < 400:
                    neighbors = np.random.choice(label_ranges['C'], k)
                else:
                    neighbors = np.random.choice(label_ranges['D'], k)
            else:
                start, end = label_ranges[lbl]
                neighbors = np.random.choice(range(start, end), k)
            
            indices.extend(neighbors)
            indptr.append(len(indices))
        
        data = np.ones(len(indices), dtype=np.float32)
        connectivities = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            min_cells_per_branch=100,
            spectral_n_init=10,
            random_state=42
        )
        assert np.isclose(score, 1.0, atol=0.05), f"Three-way failed. Expected ~1.0, got {score}"

    def test_known_f1_score():
        """Test with manually verifiable F1 score calculation"""
        G = nx.DiGraph([('A', 'B'), ('A', 'C')])
        labels = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'C', 'C'])
        # Connectivities designed for perfect clustering
        indices = [
            1,4,5,    # A0 (3)
            0,4,5,    # A1 (3)
            3,6,7,    # A2 (3)
            2,6,7,    # A3 (3)
            4,5,      # B4 (2)
            4,5,      # B5 (2)
            6,7,      # C6 (2)
            6,7       # C7 (2)
        ]
        indptr = [0, 3, 6, 9, 12, 14, 16, 18, 20]
        data = np.ones(len(indices), dtype=np.float32)
        connectivities = sparse.csr_matrix((data, indices, indptr), shape=(8,8))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            min_cells_per_branch=2,
            spectral_n_init=10,
            random_state=42,
            skip_single_branches=False
        )
        assert np.isclose(score, 1.0), f"Known F1 test failed. Expected 1.0, got {score}"

    def test_partial_misalignment():
        """Test scenario with controlled cluster assignments"""
        G = nx.DiGraph([('A', 'B'), ('A', 'C')])
        labels = np.array(['A']*4 + ['B']*2 + ['C']*2)
        # Connectivities designed for split A cells
        indices = [
            1,4,5,    # A0 (3)
            0,4,5,    # A1 (3)
            3,6,7,    # A2 (3)
            2,6,7,    # A3 (3)
            4,5,4,5,  # B4 (4)
            4,5,4,5,  # B5 (4)
            6,7,6,7,  # C6 (4)
            6,7,6,7   # C7 (4)
        ]
        indptr = [0,3,6,9,12,16,20,24,28]
        data = np.ones(len(indices), dtype=np.float32)
        connectivities = sparse.csr_matrix((data, indices, indptr), shape=(8,8))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            min_cells_per_branch=2,
            spectral_n_init=10,
            random_state=42,
            skip_single_branches=False
        )
        assert np.isclose(score, 1.0), f"Partial misalignment test failed. Expected 1.0, got {score}"

    def test_controlled_partial_score():
        """Test scenario with precisely calculable partial alignment score (0.7333)"""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C')])
        labels = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'C', 'C'])
        
        # Manually crafted connectivity matrix with controlled clustering behavior
        indices = [
            1, 2, 4, 5,    # A0 connections (A1, A2, B4, B5)
            0, 2, 4, 5,    # A1 connections (A0, A2, B4, B5)
            0, 1, 6, 7,    # A2 connections (A0, A1, C6, C7)
            3, 6, 7,       # A3 connections (A3, C6, C7)
            4, 5,          # B4 connections (B4, B5)
            4, 5,          # B5 connections (B4, B5)
            6, 7,          # C6 connections (C6, C7)
            6, 7           # C7 connections (C6, C7)
        ]
        indptr = [0,4,8,12,15,17,19,21,23]
        data = np.ones(len(indices), dtype=np.float32)
        connectivities = sparse.csr_matrix((data, indices, indptr), shape=(8,8))
        
        score = trajectory_cardinality_validation(
            given_graph=G,
            labels_array=labels,
            precomputed_embedded_connectivities=connectivities,
            validate_result=False,
            skip_single_branches=False,
            spectral_n_init=10,
            min_cells_per_branch=1,
            random_state=42
        )
        
        # Manually calculated expected F1 = (2*0.8 + 2*0.6667)/4 ≈ 0.7333
        assert np.isclose(score, 0.7333, atol=0.01), \
            f"Controlled partial score failed. Expected ~0.7333, got {score:.4f}"

    test_perfect_alignment()
    test_no_branch_points()
    test_insufficient_cells()
    test_misaligned_clusters()
    test_large_scale_validation()
    test_controlled_partial_score()
    test_known_f1_score()
    test_partial_misalignment()    
    print("All tests passed!")
