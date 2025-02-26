#!/usr/bin/env python3

import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import PCA

try:
    from sctram.evaluate._metrics.validators import validate_between_minus_plus_1 as _validator
    from sctram.evaluate._metrics.utils import Centroids, convert_scanpy_neighbors_to_indices
except ImportError:
    from validators import validate_between_minus_plus_1 as _validator
    from utils import Centroids, convert_scanpy_neighbors_to_indices


def directionality_preservation(
    given_graph: nx.DiGraph,
    inferred_embedding: np.ndarray,
    centroids: Centroids,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    validate_result: bool,
    n_neighbors: int,
    pca_components: int = 1,
) -> float:
    """Directionality preservation via local PCA and graph edge alignment.

    Computes the average cosine similarity between graph edge directions and local embedding
    principal directions. For each edge (u -> v), the direction from u's centroid to v's centroid
    is compared with the first PCA component of cells near u's centroid. Higher values indicate
    better preservation of graph-derived directions in the embedding.

    Mathematical Formulation:
        For edge (u, v):
        1. Compute Δ = centroid_v - centroid_u (normalized)
        2. Find KNN of centroid_u in embedding, fit PCA to get principal component PC1.
        3. Alignment = cos(θ) = PC1 · Δ
        Final score = mean(alignments) across all edges.

    Parameters:
        given_graph (nx.DiGraph): Directed graph with edges representing biological transitions.
        inferred_embedding (np.ndarray): Cell embeddings (n_cells x n_features).
        centroids (Centroids): Provides centroids for graph nodes via `get_centroids(nodes)`.
        precomputed_embedded_connectivities (sparse.csr_matrix): Precomputed k-nearest neighbor connectivities.
        validate_result (bool): Whether to validate the score is within [0,1].
        n_neighbors (int): Number of neighbors for local PCA (default: 50).
        pca_components (int): Number of PCA components to use (default: 1).

    Returns:
        float: Average alignment (cosine similarity) across valid edges. Range [-1,1], 
        but typically positive. Higher values (closer to 1) indicate better alignment.

    Advantages:
        - Captures directional relationships beyond pairwise distances.
        - Uses local manifold structure through PCA.
        - Automatically scales with edge density in graph.

    Limitations:
        - Assumes linear directionality in local embedding neighborhoods.
        - Sensitive to KNN and PCA parameters.
        - Requires sufficient neighbors for stable PCA estimation.
        - Edge directions in graph must reflect biological causality/pseudotime.

    Interpretation:
        Scores >0.7 suggest strong directional alignment. Negative values indicate
        anti-correlated directions (concerning). Values near 0 suggest no systematic alignment.
    """
    total_alignment = 0.0
    valid_pairs = 0
    
    embedded_neighbors = convert_scanpy_neighbors_to_indices(
        scanpy_neighbors_matrix = precomputed_embedded_connectivities,
        k = n_neighbors - 1,  # excluding itself
        include_itself = False
    )  # this should create neighbor indices except itself.

    for u, v in given_graph.edges():
        # Retrieve centroids for nodes
        centroid_u = centroids.get_single(u)
        centroid_v = centroids.get_single(v)

        # Compute graph direction vector
        graph_direction = centroid_v - centroid_u
        norm = np.linalg.norm(graph_direction)
        if norm == 0:
            raise ValueError("Edges with zero direction")

        # Find local cells around centroid_u
        local_cells = inferred_embedding[embedded_neighbors[0]]

        if len(local_cells) < 2:
            raise ValueError("insufficient cells for PCA")

        # Compute PCA direction
        pca = PCA(n_components=pca_components).fit(local_cells)
        if pca.components_.size == 0:
            raise ValueError("PCA issue")
        embedding_direction = pca.components_[0]

        # Normalize graph direction
        graph_direction_norm = graph_direction / norm

        # Calculate alignment (cosine similarity)
        alignment = np.dot(embedding_direction, graph_direction_norm)
        total_alignment += alignment
        valid_pairs += 1

    if valid_pairs == 0:
        raise ValueError("No valid edges for directionality preservation calculation.")

    score = total_alignment / valid_pairs

    if validate_result:
        _validator(score=score)

    return score


if __name__ == "__main__":
    
    class MockCentroids:
        def __init__(self, centroids_dict):
            self.centroids = {i: np.array(j) for i, j in centroids_dict.items()}
            
        def get_single(self, node):
            return self.centroids[node]

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

    def test_perfect_alignment():
        """Test perfect alignment (score=1.0)"""
        G = nx.DiGraph()
        G.add_edge('u', 'v')
        
        n_cells = 50
        inferred_embedding = np.zeros((n_cells, 2))
        inferred_embedding[0] = [0.0, 0.0]  # Centroid_u cell
        for i in range(1, n_cells):
            inferred_embedding[i] = [i*0.02, 0.0]  # Perfect x-axis alignment
        
        # Create valid neighbors matrix with correct format
        precomputed_embedded_connectivities = create_valid_scanpy_neighbors(
            n_cells=n_cells, k=49, seed_row=0
        )
        
        centroids = MockCentroids({'u': inferred_embedding[0], 'v': [1.0, 0.0]})
        
        score = directionality_preservation(
            given_graph=G,
            inferred_embedding=inferred_embedding,
            centroids=centroids,
            precomputed_embedded_connectivities=precomputed_embedded_connectivities,
            validate_result=True,
            n_neighbors=50
        )
        assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    def test_orthogonal_directions():
        """Test orthogonal directions (score≈0)"""
        G = nx.DiGraph()
        G.add_edge('u', 'v')
        
        n_cells = 50
        inferred_embedding = np.zeros((n_cells, 2))
        inferred_embedding[0] = [0.0, 0.0]
        for i in range(1, n_cells):
            inferred_embedding[i] = [0.0, i*0.02]  # Y-axis direction
        
        precomputed_embedded_connectivities = create_valid_scanpy_neighbors(
            n_cells=n_cells, k=49, seed_row=0
        )
        
        centroids = MockCentroids({'u': [0.0, 0.0], 'v': [1.0, 0.0]})
        
        score = directionality_preservation(
            G, inferred_embedding, centroids, precomputed_embedded_connectivities, 
            validate_result=True, n_neighbors=50
        )
        assert abs(score) < 0.01, f"Expected ~0, got {score}"

    def test_large_realistic_dataset():
        """Large structured dataset (score≈1.0)"""
        G = nx.DiGraph()
        nodes = ['u'] + [f'v{i}' for i in range(4)]
        for node in nodes[1:]:
            G.add_edge(nodes[0], node)
        
        n_cells = 1000
        inferred_embedding = np.zeros((n_cells, 2))
        inferred_embedding[:, 0] = np.linspace(0, 10, n_cells)  # Linear x-axis
        
        # Create neighbors matrix with first 50 cells as neighbors for centroid_u
        precomputed_embedded_connectivities = create_valid_scanpy_neighbors(
            n_cells=n_cells, k=49, seed_row=0
        )
        
        centroids_dict = {'u': [0.0, 0.0]}
        for i, node in enumerate(nodes[1:], 1):
            centroids_dict[node] = [i*2.5, 0.0]
        centroids = MockCentroids(centroids_dict)
        
        score = directionality_preservation(
            G, inferred_embedding, centroids, precomputed_embedded_connectivities,
            validate_result=True, n_neighbors=50
        )
        assert np.isclose(score, 1.0, atol=0.01), f"Expected ~1.0, got {score}"

    def test_anti_correlated_directions():
        """Test anti-correlated directions yield score=-1.0"""
        G = nx.DiGraph()
        G.add_edge('v', 'u')  # Reverse edge direction
        
        n_cells = 50
        inferred_embedding = np.zeros((n_cells, 2))
        inferred_embedding[0] = [1.0, 0.0]  # Centroid_v
        for i in range(1, n_cells):
            inferred_embedding[i] = [1.0 - i*0.02, 0.0]  # Points left of centroid_v
        
        precomputed_embedded_connectivities = create_valid_scanpy_neighbors(
            n_cells=n_cells, k=49, seed_row=0
        )
        
        centroids = MockCentroids({'v': [1.0, 0.0], 'u': [0.0, 0.0]})
        
        score = directionality_preservation(
            G, inferred_embedding, centroids, precomputed_embedded_connectivities,
            validate_result=True, n_neighbors=50
        )
        assert np.isclose(score, -1.0, atol=0.01), f"Expected -1.0, got {score}"

    def test_random_directions_average_near_zero():
        """Test random embeddings yield average alignment ≈0"""
        np.random.seed(42)
        G = nx.DiGraph()
        G.add_edges_from([('u', 'v'), ('u', 'w'), ('u', 'x')])
        
        n_cells = 500
        inferred_embedding = np.random.randn(n_cells, 50)  # High-dim random
        
        precomputed_embedded_connectivities = create_valid_scanpy_neighbors(
            n_cells=n_cells, k=49, seed_row=0
        )
        
        centroids_dict = {'u': np.zeros(50)}
        for node in ['v', 'w', 'x']:
            centroids_dict[node] = np.random.randn(50)
        centroids = MockCentroids(centroids_dict)
        
        score = directionality_preservation(
            G, inferred_embedding, centroids, precomputed_embedded_connectivities,
            validate_result=True, n_neighbors=50
        )
        assert abs(score) < 0.1, f"Expected |score| < 0.1, got {score}"

    def test_multiple_edges_varying_alignment():
        """Test mixture of alignment directions"""
        G = nx.DiGraph()
        G.add_edges_from([('u', 'v1'), ('u', 'v2'), ('u', 'v3')])
        
        n_cells = 50
        inferred_embedding = np.zeros((n_cells, 2))
        inferred_embedding[:, 0] = np.linspace(0, 1, n_cells)  # All x-aligned
        
        precomputed_embedded_connectivities = create_valid_scanpy_neighbors(
            n_cells=n_cells, k=49, seed_row=0
        )
        
        centroids = MockCentroids({
            'u': [0.0, 0.0],
            'v1': [1.0, 0.0],  # Aligned
            'v2': [0.0, 1.0],  # Orthogonal
            'v3': [-1.0, 0.0]  # Anti-aligned
        })
        
        score = directionality_preservation(
            G, inferred_embedding, centroids, precomputed_embedded_connectivities,
            validate_result=True, n_neighbors=50
        )
        expected = (1.0 + 0.0 + (-1.0)) / 3
        assert np.isclose(score, expected, atol=0.01), f"Expected {expected}, got {score}"

    def test_minimal_valid_case():
        """Test minimal valid configuration with exact neighbors"""
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C')])
        
        # 5 cells: 1 centroid + 2 neighbors for A, 1 centroid each for B and C
        inferred_embedding = np.array([
            [0.0, 0.0],  # A's centroid (row 0)
            [0.1, 0.0],  # A's neighbor 1 (row 1)
            [0.2, 0.0],  # A's neighbor 2 (row 2)
            [1.0, 0.0],  # B's centroid (row 3)
            [0.0, 1.0],  # C's centroid (row 4)
        ])
        
        # Create valid neighbor matrix with k=2 neighbors per row (n_neighbors=3)
        precomputed_embedded_connectivities = create_valid_scanpy_neighbors(
            n_cells=5, k=2, seed_row=0
        )
        
        centroids = MockCentroids({
            'A': [0.0, 0.0],
            'B': [1.0, 0.0],
            'C': [0.0, 1.0]
        })
        
        score = directionality_preservation(
            G, inferred_embedding, centroids, precomputed_embedded_connectivities,
            validate_result=True, n_neighbors=3, pca_components=1
        )
        
        # Expected alignments:
        # A->B: PCA of A's neighbors (rows 1-2) is x-axis → alignment = 1.0
        # A->C: PCA direction (x-axis) vs graph direction (y-axis) → alignment = 0.0
        expected = (1.0 + 0.0) / 2
        assert np.isclose(score, expected, atol=0.01), f"Expected {expected}, got {score}"

    
    test_perfect_alignment()
    test_orthogonal_directions()
    test_large_realistic_dataset()
    test_anti_correlated_directions()
    test_random_directions_average_near_zero()
    test_multiple_edges_varying_alignment()
    test_minimal_valid_case()
    print("All tests passed!")
    
    
    