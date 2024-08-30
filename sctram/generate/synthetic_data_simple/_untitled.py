#!/usr/bin/env python3

import numpy as np
import logging
from collections.abc import Iterable
from typing import Any, Optional, Union, List

# Logger
logger = logging.getLogger(name="synthetic_data_simple")

class SyntheticDataSimple:
    """_summary_."""  # TODO
    
    def __init__(
            self, 
            matrix: np.ndarray, 
            num_features: int = 20, 
            num_data_points: Union[int, List[int], np.ndarray]=10000, 
            centroid_method: str = "MDS",
            cluster_separation: float = 1.0, 
            randomness: float = 0.5, 
            outlier_ratio: float = 0.01,
            random_state: int = 0
        ) -> None:
        self.matrix = matrix
        self.num_features = num_features
        self.num_data_points = num_data_points
        self.cluster_separation = cluster_separation
        self.randomness = randomness
        self.outlier_ratio = outlier_ratio
        
        self.centroid_method = centroid_method
        self.centroids = self._create_centroids(self.centroid_method)
        
    def _create_centroids(self, centroid_method):
        centroid_calculator = CentroidCalculator(centroid_method)
        
        return self._create_centroids_laplacian()
        
        
        
        
        
        
        
        
        
        
        
        
                    
    def generate_data(self, start_node, num_steps):
        """
        Generates data by randomly walking through the graph, starting from a given node.
        :param start_node: The starting node for data generation.
        :param num_steps: Number of steps to simulate in the graph.
        :return: A list of visited nodes.
        """
        current_node = start_node
        visited_nodes = [current_node]
        for _ in range(num_steps):
            current_node = self._choose_next_node(current_node)
            visited_nodes.append(current_node)
        return visited_nodes

    def initialize_centers(self):
        pass
    
    def _choose_next_node(self, current_node):
        """
        Chooses the next node to visit based on the edge weights from the current node.
        :param current_node: The current node in the graph.
        :return: The next node chosen based on the weights.
        """
        probabilities = self.matrix[current_node] / np.sum(self.matrix[current_node])
        return np.random.choice(self.size, p=probabilities)

    def add_randomness(self, scale=0.1):
        """
        Adds randomness to the edge weights to introduce variability.
        :param scale: Scale of the randomness to be added.
        """
        noise = np.random.rand(self.size, self.size) * scale
        self.matrix += noise

    def set_outliers(self, number_of_outliers, outlier_weight=10):
        """
        Introduces outliers by randomly setting some edge weights to be significantly higher.
        :param number_of_outliers: Number of outlier edges.
        :param outlier_weight: The weight of the outlier edges.
        """
        for _ in range(number_of_outliers):
            i, j = np.random.randint(0, self.size, size=2)
            self.matrix[i][j] = outlier_weight