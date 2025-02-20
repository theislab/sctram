#!/usr/bin/env python3

import numpy as np


class MetricValidationError(Exception):
    """Custom exception for metric configuration errors."""
    pass


def validate_numeric(score): 
    if not isinstance(score, (int, float, np.integer, np.floating)):
        raise TypeError("Score must be a numeric type (int, float, numpy integer, or numpy float).")
    
    
def validate_placeholder(score):
    raise NotImplementedError("Metric must include a validator.")


def validate_zero_or_positive(score):
    validate_numeric(score=score)
    if score < 0:
        raise MetricValidationError("Score must be zero or positive.")
    
def validate_inclusive_between_0_1(score):
    validate_numeric(score=score)
    if score >= 0 and score <= 1:
        raise MetricValidationError("Score must in the following range: [0, 1].")
    
    