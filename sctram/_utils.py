#!/usr/bin/env python3

import logging
import numpy as np
import networkx as nx


class Utils:

    @staticmethod
    def sget(dictionary, key, logger: logging.Logger, default=None, logger_mode="warning"):
        """Retrieves the value associated with a specified key in the given dictionary.

        If the key does not exist, a default value is returned and a warning is logged.

        Parameters:
            dictionary (dict): The dictionary from which to retrieve the value.
            key (any): The key for which to retrieve the value.
            default (any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            any: The value from the dictionary associated with the specified key, or the default value
                if the key is not found.

        Raises:
            Warning: If the default value is used, a warning is issued.
        """
        if key in dictionary:
            return dictionary[key]
        else:
            message = f"Default value {default!r} used for missing key {key!r}."
            if logger_mode=="warning":
                logger.warning(message)
            elif logger_mode=="debug":
                logger.debug(message)
            elif logger_mode=="info":
                logger.info(message)
            else:
                raise ValueError(f"Unknown 'logger_mode': {logger_mode!r}")
            return default

    @staticmethod
    def validate_adjacency_matrix(arr):
        """Validates that the provided matrix is a proper adjacency matrix.
        
        Args:
        arr (np.ndarray): The matrix to validate.
        
        Raises:
            ValueError: If the matrix is not a numpy array, not 2-dimensional, not square,
                        not symmetric, or contains invalid values (values not between the minimum
                        and maximum allowable edge weights).
        """
        
        if not isinstance(arr, np.ndarray):
            raise ValueError("The matrix must be a numpy array.")
        if arr.ndim != 2:
            raise ValueError("The matrix must be 2-dimensional.")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("The matrix must be square (same number of rows and columns).")
        if not np.allclose(arr, arr.T):
            raise ValueError("The matrix must be symmetric.")
        if (np.min(arr) < 0) or (np.max(arr) > 1):  # Adjust this range as needed
            raise ValueError("All elements in the matrix must be between 0 and 1.")

    @staticmethod
    def validate_numeric_array(arr):
        """Validates that the provided data is a proper numeric 1D numpy array.

        Args:
            arr (np.ndarray): The data array to validate.

        Raises:
            ValueError: If the data is not a numpy array, not 1-dimensional, contains NaN or Inf values,
                        has non-numeric types, or if the array contains any elements outside of a specified 
                        range (for example, beyond typical float precision limits).

        """
        if not isinstance(arr, np.ndarray):
            raise ValueError("The data must be a numpy array.")
        if arr.ndim != 1:
            raise ValueError("The data must be a 1D array.")
        if not issubclass(arr.dtype.type, np.number):
            raise ValueError("The data must contain only numeric types.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("The data contains NaN or Inf values.")
        if np.any(arr < -np.finfo(np.float64).max) or np.any(arr > np.finfo(np.float64).max):
            raise ValueError("The data contains elements outside the allowable float range.")
        
    @staticmethod
    def validate_2d_matrix(arr):
        """Validates that the provided data is a proper 2D numpy array
        
        Args:
            arr (np.ndarray): The 2D matrix to validate.

        Raises:
            ValueError: If the data is not a numpy array, not 2-dimensional, contains NaN or Inf values,
                        or includes non-numeric types.
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError("The data must be a numpy array.")
        if arr.ndim != 2:
            raise ValueError("The data must be a 2D array.")
        if not issubclass(arr.dtype.type, np.number):
            raise ValueError("All elements in the data must be numeric.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("The data contains NaN or Inf values.")

    @staticmethod
    def adjacency_graph_to_matrix(g: nx.DiGraph, nodelist_filter_and_order) -> np.ndarray:
        symetrical_graph = g.to_symetrical_multidigraph()  # to have symetrical adjacency matrices
        adj_matrix = nx.to_numpy_array(symetrical_graph, nodelist=nodelist_filter_and_order)
        return adj_matrix