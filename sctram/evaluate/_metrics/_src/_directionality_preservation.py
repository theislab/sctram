#!/usr/bin/env python3

from typing import Optional

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

try:
    from sctram.evaluate._metrics._src.utils import Centroids, convert_scanpy_neighbors_to_indices
    from sctram.evaluate._metrics._src.validators import validate_between_minus_plus_1 as _validator
except ImportError:
    from utils import Centroids, convert_scanpy_neighbors_to_indices
    from validators import validate_between_minus_plus_1 as _validator


def _find_edge_cells(u, v, inferred_embedding, centroids, k, labels_array, orthogonal_distance_coefficient=np.inf):
    """
    Find k cells from inferred_embedding that are labeled as belonging to node u and lie between
    the centroids of node u and node v. Cells are selected based on their projection along the edge
    direction, and only those within the segment defined by the centroids are considered.

    Parameters:
        u (str): Source node label.
        v (str): Destination node label.
        inferred_embedding (np.ndarray): Cell embeddings (n_cells x n_features).
        centroids (Centroids): Object that provides centroids for graph nodes.
        labels_array (np.array): Array of cell type labels, with length n.
        orthogonal_distance_coefficient (float): If infinite, it is effectively disabled.
            Cells are not only aligned with the edge direction but are also spatially close to the edge are chosen.
            This filters out cells that might lie along the projection line but are too far from the actual biological
            path (as represented by the edge between the centroids). The coefficient is chosen to determine
            distance relative to the edge lenght.
        k (int): Number of cells to return.

    Returns:
        np.array: Indices of selected cells.
    """
    centroid_u = centroids.get_single(u)
    centroid_v = centroids.get_single(v)
    edge_vector = centroid_v - centroid_u
    norm_edge = np.linalg.norm(edge_vector)
    if norm_edge == 0:
        raise ValueError(f"Edge {u}->{v} has zero length.")
    edge_direction = edge_vector / norm_edge

    # Calculate projection of each cell onto the edge direction.
    projection_scalars = np.dot(inferred_embedding - centroid_u, edge_direction)
    projection_vectors = np.outer(projection_scalars, edge_direction)
    difference_vectors = inferred_embedding - centroid_u
    orthogonal_vectors = difference_vectors - projection_vectors
    orthogonal_distances = np.linalg.norm(orthogonal_vectors, axis=1)

    # Filter cells by belonging to node u using labels_array.
    node_u_mask = labels_array == u
    # Cells' projection values are within 0 and norm_edge, and orthogonal distances less than d/2.
    orthogonal_mask = orthogonal_distances < norm_edge * orthogonal_distance_coefficient

    valid_mask = (projection_scalars > 0) & (projection_scalars < norm_edge) & orthogonal_mask & node_u_mask
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        raise ValueError(
            f"No valid cells found between centroids for edge {u}->{v} belonging "
            "to {u} within acceptable orthogonal distance."
        )

    # Choose the k cells closest to the midpoint of the edge.
    midpoint = centroid_u + edge_vector / 2
    distances = np.linalg.norm(inferred_embedding[valid_indices] - midpoint, axis=1)
    sorted_indices = valid_indices[np.argsort(distances)]
    selected = sorted_indices[:k] if len(sorted_indices) >= k else sorted_indices
    return selected


def _get_edge_local_cells(u, v, inferred_embedding, centroids, embedded_neighbors, k, labels_array):
    """
    First, find k cells that lie along the edge from centroid_u to centroid_v.
    Then, using the precomputed neighbor indices, take the union of the neighbors
    of these k cells to form the local cell set for PCA.
    """
    edge_cells = _find_edge_cells(u, v, inferred_embedding, centroids, k, labels_array)
    # print("len(edge_cells)", len(edge_cells))
    neighbor_indices = set()
    for cell_idx in edge_cells:
        # embedded_neighbors is expected to be an array-like where each element contains
        # the indices of neighbor cells for that particular cell.
        neighbor_indices.update(embedded_neighbors[cell_idx])
    neighbor_indices = np.array(list(neighbor_indices))
    local_cells = inferred_embedding[neighbor_indices]
    return local_cells


def directionality_preservation(
    given_graph: nx.DiGraph,
    inferred_embedding: np.ndarray,
    labels_array: np.ndarray,
    centroids: Centroids,
    precomputed_embedded_connectivities: sparse.csr_matrix,
    validate_result: bool,
    n_neighbors: int,
    k_cells: Optional[int] = None,
    pca_components: int = 1,
) -> float:
    """Directionality preservation via local PCA and graph edge alignment.

    For each edge (u -> v), this function:
      1. Retrieves centroids for nodes u and v.
      2. Computes the graph direction vector (from u to v).
      3. Identifies k cells along the line connecting centroid_u and centroid_v.
      4. Expands these cells by unioning their precomputed neighbors.
      5. Runs PCA on these local cells and compares the primary PCA component with the
         normalized graph direction (using cosine similarity).
      6. Averages the alignments over all edges.

    Mathematical Formulation:
        For edge (u, v):
        1. Compute Δ = centroid_v - centroid_u (normalized)
        2. Find KNN of centroid_u in embedding, fit PCA to get principal component PC1.
        3. Alignment = cos(θ) = PC1 · Δ
        Final score = mean(alignments) across all edges.

    Parameters:
        given_graph (nx.DiGraph): Directed graph with edges representing biological transitions.
        inferred_embedding (np.ndarray): Cell embeddings (n_cells x n_features).
        labels_array (np.array): Array of cell type labels, with length n.
        centroids (Centroids): Provides centroids for graph nodes via `get_centroids(nodes)`.
        precomputed_embedded_connectivities (sparse.csr_matrix): Precomputed k-nearest neighbor connectivities.
        validate_result (bool): Whether to validate the score is within [0,1].
        n_neighbors (int): Number of neighbors for local PCA (default: 50).
        k_cells (Optional[int]): Number of cells to choose along centroids. If None, n_neighbors will be used.
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
    k_cells = n_neighbors if k_cells is None else k_cells

    # Precompute neighbor indices for all cells.
    embedded_neighbors = convert_scanpy_neighbors_to_indices(
        scanpy_neighbors_matrix=precomputed_embedded_connectivities,
        k=n_neighbors - 1,  # excluding the cell itself
        include_itself=False,
    )

    for u, v in given_graph.edges():
        # Retrieve centroids for nodes u and v.
        centroid_u = centroids.get_single(u)
        centroid_v = centroids.get_single(v)

        # Compute graph direction vector and normalize it.
        graph_direction = centroid_v - centroid_u
        norm = np.linalg.norm(graph_direction)
        if norm == 0:
            raise ValueError("Edges with zero direction")
        graph_direction_norm = graph_direction / norm

        # Instead of taking neighbors around centroid_u, find k cells along the edge from u to v,
        # then get the union of their neighbors.
        local_cells = _get_edge_local_cells(
            u, v, inferred_embedding, centroids, embedded_neighbors, k_cells, labels_array
        )
        # print("len(local_cells)", len(local_cells))
        if len(local_cells) < 2:
            raise ValueError("Insufficient cells for PCA")

        # Compute PCA on the local cells.
        pca = PCA(n_components=pca_components).fit(local_cells)
        if pca.components_.size == 0:
            raise ValueError("PCA issue")
        embedding_direction = pca.components_[0]

        # Calculate alignment (cosine similarity).
        alignment = np.dot(embedding_direction, graph_direction_norm)
        total_alignment += alignment
        valid_pairs += 1

    if valid_pairs == 0:
        raise ValueError("No valid edges for directionality preservation calculation.")

    score = total_alignment / valid_pairs

    if validate_result:
        _validator(score=score)

    return score
