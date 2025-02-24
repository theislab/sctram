#!/usr/bin/env python3

import numpy as np


class MetricValidationError(Exception):
    """Custom exception for metric configuration errors."""
    pass


def validate_numeric(score): 
    if not isinstance(score, (int, float, np.integer, np.floating)):
        raise TypeError(f"Score must be a numeric type (int, float, numpy integer, or numpy float), got {type(score)!r}.")
    
    
def validate_placeholder(score):
    raise NotImplementedError(f"Metric must include a validator.")


def validate_zero_or_positive(score):
    validate_numeric(score=score)
    if score < 0:
        raise MetricValidationError(f"Score must be zero or positive, got {score !r}.")
    
def validate_inclusive_between_0_1(score):
    validate_numeric(score=score)
    if not (score >= 0 and score <= 1):
        raise MetricValidationError(f"Score must in the following range: [0, 1], got {score !r}.")
    
def validate_between_minus_plus_1(score):
    validate_numeric(score=score)
    if not (score >= -1 and score <= 1):
        raise MetricValidationError(f"Score must in the following range: [-1, 1], got {score !r}.")
    
def validate_maximum_1(score):
    validate_numeric(score=score)
    if score > 1:
        raise MetricValidationError(f"Score must be lower than 1, got {score !r}.")
