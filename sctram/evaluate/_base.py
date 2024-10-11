#!/usr/bin/env python3

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from sctram.input import InputTrajectories


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
        given_trajectory (InputTrajectories): The ground truth trajectory.
        inferred (Any): The inferred trajectory.
        subset_given (Any): Subset of the given trajectory.
        subset_inferred (Any): Subset of the inferred trajectory.
        prepared_before_subset_given (Any): Prepared given trajectory before subsetting.
        prepared_before_subset_inferred (Any): Prepared inferred trajectory before subsetting.
        prepared_after_subset_given (Any): Prepared given trajectory after subsetting.
        prepared_after_subset_inferred (Any): Prepared inferred trajectory after subsetting.
        result (Any): The result of the evaluation.
    """

    def __init__(
        self,
        method_params: Optional[Dict[str, Any]] = None,
        subset_params: Optional[Dict[str, Any]] = None,
        prepare_params_before_subset: Optional[Dict[str, Any]] = None,
        prepare_params_after_subset: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the evaluation method with common parameters.

        Args:
            method_params (Optional[Dict[str, Any]]): Parameters specific to the evaluation method.
            subset_params (Optional[Dict[str, Any]]): Parameters for subsetting the data.
            prepare_params_before_subset (Optional[Dict[str, Any]]): Parameters for preparing the data before subsetting.
            prepare_params_after_subset (Optional[Dict[str, Any]]): Parameters for preparing the data after subsetting.
        """
        self.method_params = method_params or {}
        self.subset_params = subset_params or {}
        self.prepare_params_before_subset = prepare_params_before_subset or {}
        self.prepare_params_after_subset = prepare_params_after_subset or {}

        self.logger = logging.getLogger(self.__class__.__name__)

        self.given_trajectory: Optional[InputTrajectories] = None
        self.inferred: Any = None

        self.subset_given: Any = None
        self.subset_inferred: Any = None

        self.prepared_before_subset_given: Any = None
        self.prepared_before_subset_inferred: Any = None

        self.prepared_after_subset_given: Any = None
        self.prepared_after_subset_inferred: Any = None

        self.result: Any = None

    def evaluate(
        self,
        given_trajectory: InputTrajectories,
        inferred: Any,
    ) -> Any:
        """Evaluates the inferred trajectory against the given trajectory.

        This method follows these steps:
            1. Verifies the given and inferred trajectories.
            2. Prepares the trajectories before subsetting based on `prepare_params_before_subset`.
            3. Optionally subsets the trajectories based on `subset_params`.
            4. Prepares the trajectories after subsetting based on `prepare_params_after_subset`.
            5. Performs the evaluation calculation.
            6. Returns the result.

        Args:
            given_trajectory (InputTrajectories): The ground truth trajectory.
            inferred (Any): The inferred trajectory, format depends on the subclass implementation.

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
            self.inferred = self._verify_inferred_trajectory(inferred)

            # Prepare trajectories before subsetting
            if self.prepare_params_before_subset:
                self.logger.debug("Preparing trajectories before subsetting.")
                self.prepared_before_subset_given, self.prepared_before_subset_inferred = self._prepare_before_subset(
                    self.given_trajectory, self.inferred
                )
            else:
                self.prepared_before_subset_given = self.given_trajectory
                self.prepared_before_subset_inferred = self.inferred

            # Subset the trajectories if needed
            if self.subset_params:
                self.logger.debug("Subsetting trajectories.")
                self.subset_given, self.subset_inferred = self._subset(
                    self.prepared_before_subset_given, self.prepared_before_subset_inferred
                )
            else:
                self.subset_given = self.prepared_before_subset_given
                self.subset_inferred = self.prepared_before_subset_inferred

            # Prepare trajectories after subsetting
            if self.prepare_params_after_subset:
                self.logger.debug("Preparing trajectories after subsetting.")
                self.prepared_after_subset_given, self.prepared_after_subset_inferred = self._prepare_after_subset(
                    self.subset_given, self.subset_inferred
                )
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

    def _verify_given_trajectory(self, given_trajectory: InputTrajectories) -> InputTrajectories:
        """Verifies the given trajectory.

        Args:
            given_trajectory (InputTrajectories): The given trajectory.

        Returns:
            InputTrajectories: The verified given trajectory.

        Raises:
            ValueError: If the given trajectory is not a networkx.MultiDiGraph.
        """
        if not isinstance(given_trajectory, InputTrajectories):
            raise ValueError("Given trajectory must be a networkx.MultiDiGraph.")
        return given_trajectory

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
        """
        pass

    @abstractmethod
    def _prepare_before_subset(self, given_trajectory: Any, inferred_trajectory: Any) -> Tuple[Any, Any]:
        """Prepares the trajectories before subsetting.

        This method must be implemented in subclasses to handle preparation specific to the data formats before subsetting.

        Args:
            given_trajectory (Any): The given trajectory.
            inferred_trajectory (Any): The inferred trajectory.

        Returns:
            Tuple[Any, Any]: The prepared given and inferred trajectories before subsetting.
        """
        pass

    @abstractmethod
    def _subset(self, given_trajectory: Any, inferred_trajectory: Any) -> Tuple[Any, Any]:
        """Subsets the trajectories based on `subset_params`.

        This method must be implemented in subclasses to handle subsetting specific to the data formats.

        Args:
            given_trajectory (Any): The prepared given trajectory before subsetting.
            inferred_trajectory (Any): The prepared inferred trajectory before subsetting.

        Returns:
            Tuple[Any, Any]: The subsetted given and inferred trajectories.
        """
        pass

    @abstractmethod
    def _prepare_after_subset(self, subset_given: Any, subset_inferred: Any) -> Tuple[Any, Any]:
        """Prepares the trajectories after subsetting.

        This method must be implemented in subclasses to handle preparation specific to the data formats after subsetting.

        Args:
            subset_given (Any): The subsetted given trajectory.
            subset_inferred (Any): The subsetted inferred trajectory.

        Returns:
            Tuple[Any, Any]: The prepared given and inferred trajectories after subsetting.
        """
        pass

    @abstractmethod
    def _calculate(self):
        """Performs the specific evaluation calculation.

        This method must be implemented in subclasses to perform the evaluation and store the result in `self.result`.
        """
        pass

    @abstractmethod
    def get_result(self) -> Any:
        """Retrieves the result of the evaluation.

        Returns:
            Any: The result of the evaluation.

        Raises:
            ValueError: If the result is not available.
        """
        if self.result is None:
            raise ValueError("No result available. Have you run the evaluation?")
        return self.result
