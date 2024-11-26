#!/usr/bin/env python3

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sctram._constants import evaluate_metrics_key, sctram_operate_key
from sctram.input import InputTrajectory


class EvaluationBase(ABC):
    """Abstract base class for trajectory evaluation methods.

    This class defines the common interface and workflow for evaluating inferred trajectories against a given trajectory.
    It handles the verification of inputs, optional subsetting based on cell labels or other criteria, preparation of
    data for evaluation before and after subsetting, and the execution of the evaluation itself.

    Subclasses should implement the specific verification, subsetting, preparation, and calculation methods, as these
    can vary depending on the format of the inferred trajectory (e.g., adjacency matrix, pseudotime vector, networkx graph).

    Attributes:
        method_params (Dict[str, Any]): Parameters specific to the evaluation method.
        subset_params (Dict[str, Any]): Parameters for subsetting the data.
        prepare_params_before_subset (Dict[str, Any]): Parameters for preparing the data before subsetting.
        prepare_params_after_subset (Dict[str, Any]): Parameters for preparing the data after subsetting.
        logger (logging.Logger): Logger for the class.
        given_trajectory (InputTrajectory): The ground truth trajectory.
        inferred (Any): The inferred trajectory.
        subset_given (Any): Subset of the given trajectory.
        subset_inferred (Any): Subset of the inferred trajectory.
        prepared_before_subset_given (Any): Prepared given trajectory before subsetting.
        prepared_before_subset_inferred (Any): Prepared inferred trajectory before subsetting.
        prepared_after_subset_given (Any): Prepared given trajectory after subsetting.
        prepared_after_subset_inferred (Any): Prepared inferred trajectory after subsetting.
        result (Any): The result of the evaluation.
    """

    available_metrics: List[str]  # This is needed to be completed in metricsmixin classes.

    def __init__(
        self,
        method_params: Dict[str, Any],
        subset_params: Optional[Dict[str, Any]] = None,
        prepare_params_before_subset: Optional[Dict[str, Any]] = None,
        prepare_params_after_subset: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the evaluation method with common parameters.

        Args:
            method_params (Dict[str, Any]): Parameters specific to the evaluation method.
            subset_params (Optional[Dict[str, Any]]): Parameters for subsetting the data.
            prepare_params_before_subset (Optional[Dict[str, Any]]): Parameters for preparing the data before subsetting.
            prepare_params_after_subset (Optional[Dict[str, Any]]): Parameters for preparing the data after subsetting.
        """
        self.method_params, self.metrics = self._verify_method_params(method_params=method_params)
        self.subset_params = self._params_variable_prepare(subset_params)
        self.prepare_params_before_subset = self._params_variable_prepare(prepare_params_before_subset)
        self.prepare_params_after_subset = self._params_variable_prepare(prepare_params_after_subset)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Labels given in `evaluate` method.
        self.labels: Any = None  # New attribute to store labels
        self.subset_labels: Any = None

        # Unprocessed inputs to `evaluate` method.
        self.given_trajectory: Any = None
        self.inferred_trajectory: Any = None

        # In general, either one of the prepare methods are used but both are kept in
        # this parent class for code consistency.
        self.prepared_before_subset_given: Any = None
        self.prepared_before_subset_inferred: Any = None

        self.subset_given: Any = None
        self.subset_inferred: Any = None

        self.prepared_after_subset_given: Any = None
        self.prepared_after_subset_inferred: Any = None

        # The results are kept as a dict.
        self.result: Dict[str, Union[int, float]] = dict()

    def _verify_method_params(self, method_params: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
        if method_params is None or not isinstance(method_params, dict):
            raise ValueError("`method_params` should be a dict.")
        if evaluate_metrics_key not in method_params:
            raise ValueError(f"`method_params` must contain {evaluate_metrics_key!r} key.")

        metrics = method_params[evaluate_metrics_key]
        if not isinstance(metrics, list):
            raise ValueError(f"metrics_key {evaluate_metrics_key!r} should be a list of metric names.")

        if self.available_metrics is None or len(self.available_metrics) == 0:
            raise NotImplementedError(f"Subclass {self.__class__.__name__!r} should define `available_metrics`.")

        for metric in metrics:
            if metric not in self.available_metrics:
                raise ValueError(f"Undefined metric: {metric!r}")

        return method_params, metrics

    def _params_variable_prepare(self, params_variable):
        if params_variable is None:
            params_variable = dict()

        if not isinstance(params_variable, dict):
            raise ValueError(
                f"Expected a dictionary for params_variable, but received type: {type(params_variable).__name__}."
            )

        for key in params_variable:
            if not isinstance(key, str):
                raise ValueError(
                    f"All keys in the params_variable must be strings, found key of type: {type(key).__name__}."
                )
            if key == sctram_operate_key:
                raise ValueError(f"The key {sctram_operate_key!r} is not allowed in params_variable")

        if len(params_variable) != 0:
            params_variable[sctram_operate_key] = True
        else:
            params_variable[sctram_operate_key] = False

        return params_variable

    def evaluate(
        self,
        given_trajectory: InputTrajectory,
        inferred_trajectory: Any,
        labels: np.ndarray,
    ) -> Any:
        """Evaluates the inferred trajectory against the given trajectory.

        This method follows these steps:
            1. Verifies the given and inferred trajectories.
            2. Verifies the labels.
            3. Prepares the trajectories before subsetting based on `prepare_params_before_subset`.
            4. Optionally subsets the trajectories based on `subset_params`.
            5. Prepares the trajectories after subsetting based on `prepare_params_after_subset`.
            6. Compares the trajectories with labels.
            7. Performs the evaluation calculation.
            8. Returns the result.

        Args:
            given_trajectory (InputTrajectory): The ground truth trajectory.
            inferred_trajectory (Any): The inferred trajectory, format depends on the subclass implementation.
            labels (np.ndarray): A 1D NumPy array of string labels.

        Returns:
            Any: The result of the evaluation.

        Raises:
            RuntimeError: If an error occurs during evaluation.
        """
        try:
            self.logger.info(f"Starting evaluation with {self.__class__.__name__}.")

            # Verify the trajectories
            self.logger.debug("Verifying given trajectory.")
            self.given_trajectory = self._verify_given_trajectory(given_trajectory)

            self.logger.debug("Verifying inferred trajectory.")
            self.inferred_trajectory = self._verify_inferred_trajectory(inferred_trajectory)

            # Verify labels
            self.logger.debug("Verifying labels.")
            self.labels = self._verify_labels(labels)
            self._verify_labels_data_specific()

            # Prepare trajectories before subsetting
            if self.prepare_params_before_subset[sctram_operate_key]:
                self.logger.debug("Preparing trajectories before subsetting.")
                self.prepared_before_subset_given, self.prepared_before_subset_inferred = self._prepare_before_subset()
            else:
                self.prepared_before_subset_given = self.given_trajectory
                self.prepared_before_subset_inferred = self.inferred_trajectory

            # Subset the trajectories if needed
            if self.subset_params[sctram_operate_key]:
                self.logger.debug("Subsetting trajectories.")
                self.subset_given, self.subset_inferred, self.subset_labels = self._subset()
            else:
                self.subset_given = self.prepared_before_subset_given
                self.subset_inferred = self.prepared_before_subset_inferred
                self.subset_labels = self.labels

            # Prepare trajectories after subsetting
            if self.prepare_params_after_subset[sctram_operate_key]:
                self.logger.debug("Preparing trajectories after subsetting.")
                self.prepared_after_subset_given, self.prepared_after_subset_inferred = self._prepare_after_subset()
            else:
                self.prepared_after_subset_given = self.subset_given
                self.prepared_after_subset_inferred = self.subset_inferred

            # Perform the evaluation calculation
            self.logger.debug("Performing evaluation calculation.")
            self._calculate()

            self.logger.info("Evaluation completed successfully.")
            return self.get_result()

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise RuntimeError(f"Error during evaluation: {e}") from e

    def _verify_given_trajectory(self, given_trajectory: InputTrajectory) -> InputTrajectory:
        """Verifies the given trajectory.

        Args:
            given_trajectory (InputTrajectory): The given trajectory.

        Returns:
            InputTrajectory: The verified given trajectory.

        Raises:
            ValueError: If the given trajectory is not a InputTrajectory.
        """
        if not isinstance(given_trajectory, InputTrajectory):
            raise ValueError("Given trajectory must be a InputTrajectory instance.")
        return given_trajectory

    def _verify_labels(self, labels: np.ndarray) -> np.ndarray:
        """Verifies that labels is a 1D NumPy array of strings.

        Args:
            labels (np.ndarray): The labels to verify.

        Returns:
            np.ndarray: The verified labels.

        Raises:
            ValueError: If labels is not a 1D NumPy array of strings.
        """
        if not isinstance(labels, np.ndarray):
            raise ValueError("Labels must be a NumPy array.")
        if labels.ndim != 1:
            raise ValueError("Labels array must be 1-dimensional.")
        if labels.dtype.type is not np.str_ and labels.dtype.type is not np.object_:
            # np.str_ covers fixed-length strings, np.object_ can include Python strings
            raise ValueError("Labels array must contain strings.")

        return labels

    @abstractmethod
    def _verify_labels_data_specific(self):
        """Verifies that labels is consistent with input trajectory and/or inferred trajectory.

        This method must be implemented in subclasses to handle the specific format.

        Raises:
            NotImplementedError: as this is abstractmethod.
        """
        raise NotImplementedError

    @abstractmethod
    def _verify_inferred_trajectory(self, inferred: Any) -> Any:
        """Verifies the inferred trajectory.

        This method must be implemented in subclasses to handle the specific format of the inferred trajectory.

        Args:
            inferred (Any): The inferred trajectory.

        Returns:
            Any: The verified inferred trajectory.

        Raises:
            ValueError: If the inferred trajectory is invalid.
            NotImplementedError: as this is abstractmethod.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_before_subset(self) -> Tuple[Any, Any]:
        """Prepares the trajectories before subsetting.

        This method must be implemented in subclasses to handle preparation
        specific to the data formats before subsetting.

        Returns:
            Tuple[Any, Any]: The prepared given and inferred trajectories before subsetting.

        Raises:
            NotImplementedError: as this is abstractmethod.
        """
        raise NotImplementedError

    @abstractmethod
    def _subset(self) -> Tuple[Any, Any, Any]:
        """Subsets the trajectories based on `subset_params`.

        This method must be implemented in subclasses to handle subsetting specific to the data formats.

        Returns:
            Tuple[Any, Any]: The subsetted given and inferred trajectories.

        Raises:
            NotImplementedError: as this is abstractmethod.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_after_subset(self) -> Tuple[Any, Any]:
        """Prepares the trajectories after subsetting.

        This method must be implemented in subclasses to handle preparation specific
        to the data formats after subsetting.

        Returns:
            Tuple[Any, Any]: The prepared given and inferred trajectories after subsetting.

        Raises:
            NotImplementedError: as this is abstractmethod.
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate(self):
        """Performs the specific evaluation calculation.

        This method must be implemented in subclasses to perform the evaluation and store the result in `self.result`.

        Raises:
            NotImplementedError: as this is abstractmethod.
        """
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        """Retrieves the result of the evaluation.

        Raises:
            NotImplementedError: as this is abstractmethod.
        """
        raise NotImplementedError
