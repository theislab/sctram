#!/usr/bin/env python3

from loguru._logger import Logger
from typing import Any, Dict, List

import numpy as np


class MetricsMixinBase:
    """Declarations for consistency and mypy."""

    # Declare expected attributes with type annotations
    prepared_after_subset_given: np.ndarray
    prepared_after_subset_inferred: np.ndarray
    result: Dict[str, Any]
    metrics: List[str]
    method_params: Dict[str, Any]
    logger: Logger
