#!/usr/bin/env python3

import numpy as np
from scipy.sparse.csgraph import laplacian, shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS, Isomap, SpectralEmbedding


class LaplacianEigenmaps:
    def __init__(self, n_components):
        """Initializes the Laplacian Eigenmaps model.

        Args:
            n_components (int): The number of dimensions for the reduced space.
        """
        self.n_components = n_components

    def fit_transform(self, X):
        """Compute the low-dimensional embedding."""
        # Compute the normalized Laplacian matrix
        laplacian_matrix = laplacian(X, normed=True)
        # Compute the first `n_components + 1` smallest eigenvalues and eigenvectors
        _, eigvecs = eigsh(laplacian_matrix, k=self.n_components + 1, which="SM")
        # Discard the first eigenvector (corresponding to the zero eigenvalue)
        embedding = eigvecs[:, 1 : self.n_components + 1]
        return embedding


class CentroidCalculator:
    """Calculates centroids using various dimensionality reduction methods."""

    def __init__(
        self, centroid_method: str, adjacency_matrix: np.ndarray, num_features: int, random_state: int
    ) -> None:
        """Initializes the CentroidCalculator.

        Args:
            centroid_method (str): The method used for centroid calculation. Choices are 'mds', 'spectral', 'isomap', 'kernel_pca', and 'laplacian'.
            adjacency_matrix (np.ndarray): The adjacency matrix representing the graph (or any square matrix).
            num_features (int): The number of dimensions for the reduced space.
            random_state (int): A seed value to ensure reproducibility.
        """
        self.centroid_method = centroid_method
        self.adjacency_matrix = adjacency_matrix.copy()
        self.num_features = num_features
        self.random_state = random_state

        if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise ValueError("The adjacency matrix must be a square matrix.")

        if not np.all(np.diag(self.adjacency_matrix) == 0):
            raise ValueError("All diagonal elements of the adjacency matrix must be zero.")

        if not np.allclose(self.adjacency_matrix, self.adjacency_matrix.T):
            raise ValueError("Adjacency matrix must be symmetrical.")

    def get(self):
        """Create k-dimensional centroids using the specified method based on the edge weights.

        Each method has specific characteristics and use cases:

        'mds': Multidimensional Scaling (MDS)
            - Preserves the pairwise distance between nodes as well as possible in the lower-dimensional space.
            - Useful for visualizing similarity or dissimilarity data, such as in social networks where edges represent similarity in attributes.

        'spectral': Spectral Embedding
            - Utilizes the graph Laplacian to perform a low-dimensional embedding, emphasizing the clustering of nodes in the graph based on their connectivity.
            - Best used for graphs where you expect to find a clear clustering structure or communities, such as in citation networks or community detection tasks.

        'isomap': Isometric Mapping
            - Seeks to preserve the geodesic distances in the original high-dimensional space. It is effectively an MDS that uses geodesic distances instead of Euclidean ones.
            - Ideal for data where global manifold structure is important, such as in data that lie on a curved manifold, offering a perspective that maintains global properties.

        'kernel_pca': Kernel PCA
            - Extends PCA to nonlinear dimensionality reduction through the use of kernels, capable of capturing complex structures in the data.
            - Suitable for data where nonlinear relationships are significant, such as in image processing or any complex network structures where linear projections fail to capture the essence of data relations.

        'laplacian': Laplacian Eigenmaps
            - Focuses on preserving local node relationships and is particularly sensitive to the choice of neighborhood or the graph construction.
            - Effective for data where local relationships are more significant than global relationships, such as in spectral clustering or when data has a very irregular structure.

        Returns:
            np.ndarray: The centroids in the reduced space, capturing essential relationships as specified by the method.
        """
        if self.centroid_method == "mds":
            return self._create_centroids_mds()
        elif self.centroid_method == "spectral":
            return self._create_centroids_spectral()
        elif self.centroid_method == "isomap":
            return self._create_centroids_isomap()
        elif self.centroid_method == "kernel_pca":
            return self._create_centroids_kernel_pca()
        elif self.centroid_method == "laplacian":
            return self._create_centroids_laplacian()
        else:
            raise ValueError(f"Unknown centroid method: {self.centroid_method}")

    def _compute_distance_matrix(self):
        """Compute the distance matrix from the adjacency (similarity) matrix."""
        # Convert similarities to distances: higher similarities correspond to shorter distances
        epsilon = 1e-8  # Small value to prevent division by zero
        similarity_matrix = self.adjacency_matrix + epsilon
        # Invert similarities to get distances
        distance_matrix = 1 / similarity_matrix
        # Now compute the shortest path distances
        distance_matrix = shortest_path(csgraph=distance_matrix, method="D", directed=False, unweighted=False)
        return distance_matrix

    def _create_centroids_mds(self) -> np.ndarray:
        """Multidimensional Scaling (MDS)."""
        distance_matrix = self._compute_distance_matrix()
        mds = MDS(
            n_components=self.num_features,
            dissimilarity="precomputed",
            random_state=self.random_state,
        )
        centroids = mds.fit_transform(distance_matrix)
        return centroids

    def _create_centroids_spectral(self) -> np.ndarray:
        """Spectral Embedding."""
        embedding = SpectralEmbedding(
            n_components=self.num_features,
            affinity="precomputed",
            random_state=self.random_state,
        )
        centroids = embedding.fit_transform(self.adjacency_matrix)
        return centroids

    def _create_centroids_isomap(self) -> np.ndarray:
        """Isomap."""
        distance_matrix = self._compute_distance_matrix()
        isomap = Isomap(n_components=self.num_features, metric="precomputed")
        centroids = isomap.fit_transform(distance_matrix)
        return centroids

    def _create_centroids_kernel_pca(self, gamma=None) -> np.ndarray:
        """Kernel PCA."""
        # Compute the kernel matrix using RBF kernel based on distances
        distance_matrix = self._compute_distance_matrix()
        if gamma is None:
            # Set gamma as the inverse of the median of the distances (excluding infinities)
            finite_distances = distance_matrix[np.isfinite(distance_matrix)]
            gamma = 1 / np.median(finite_distances)
        # Compute RBF kernel from distances
        kernel_matrix = np.exp(-gamma * distance_matrix**2)
        kernel_pca = KernelPCA(
            n_components=self.num_features,
            kernel="precomputed",
            random_state=self.random_state,
        )
        centroids = kernel_pca.fit_transform(kernel_matrix)
        return centroids

    def _create_centroids_laplacian(self) -> np.ndarray:
        """Laplacian Eigenmaps."""
        laplacian_eigenmaps = LaplacianEigenmaps(n_components=self.num_features)
        centroids = laplacian_eigenmaps.fit_transform(self.adjacency_matrix)
        return centroids
