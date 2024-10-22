#!/usr/bin/env python3

from typing import Dict, Any, List
import logging
import numpy as np

class MetricsMixinBase:
    """Declarations for consistency and mypy."""

    # Declare expected attributes with type annotations
    prepared_after_subset_given: np.ndarray
    prepared_after_subset_inferred: np.ndarray
    result: Dict[str, Any]
    metrics: List[str]
    method_params: Dict[str, Any]
    logger: logging.Logger