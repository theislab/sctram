#!/usr/bin/env python3

from loguru import logger
from typing import Any, Dict

import os
import yaml 
from sctram.api._class_mapping import *

# Load defaults from defaults.yaml
DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), "_defaults.yaml")

def load_default_metrics(yaml_path: str) -> Dict[str, Any]:
    """Reads the default metric combinations from a YAML file. Also performs detailed checks to ensure the structure is correct."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        defaults = yaml.safe_load(f)
    
    logger.info("Loaded default metrics from {}", yaml_path)
    
    # Check top-level structure
    if not isinstance(defaults, dict):
        raise ValueError("The YAML file must contain a dictionary at the top level.")
    
    if "combinations" not in defaults:
        raise ValueError("The YAML file must contain a 'combinations' key.")
    
    if not isinstance(defaults["combinations"], list):
        raise ValueError("The 'combinations' key must be associated with a list.")
    
    # Iterate over each combination and validate structure
    for idx, combo in enumerate(defaults["combinations"]):
        if not isinstance(combo, dict):
            raise ValueError(f"Combination entry at index {idx} is not a dictionary.")
        
        # Check required keys
        required_keys = {"evaluate", "infer", "metrics", "description"}
        missing = required_keys - combo.keys()
        if missing:
            raise ValueError(f"Combination entry at index {idx} is missing keys: {missing}")
        
        # Validate evaluation class name
        eval_class = combo["evaluate"]
        if eval_class not in EVALUATION_ALLOWED_INFER:
            allowed = list(EVALUATION_ALLOWED_INFER.keys())
            raise ValueError(f"Invalid evaluate class '{eval_class}' at index {idx}. "
                             f"Allowed values are: {allowed}")
        
        # Validate inference class name for the evaluation
        infer_class = combo["infer"]
        allowed_infer = EVALUATION_ALLOWED_INFER[eval_class]
        if infer_class not in allowed_infer:
            raise ValueError(f"In combination at index {idx}, the infer class '{infer_class}' is not allowed for evaluation class '{eval_class}'. "
                             f"Allowed inference classes are: {allowed_infer}")
        
        # Validate metrics is a list
        if not isinstance(combo["metrics"], list):
            raise ValueError(f"The 'metrics' value in combination at index {idx} must be a list.")
        
        # Optionally, check that each metric is a string.
        for metric in combo["metrics"]:
            if not isinstance(metric, str):
                raise ValueError(f"Each metric in the 'metrics' list must be a string. Found {type(metric)} in combination at index {idx}.")
            
            if not metric in EVALUATION_METHODS[combo["evaluate"]].available_metrics:
                raise ValueError(f"Metric {metric!r} is not available in {combo['evaluate']!r} class.")
        
        # Validate description
        if not isinstance(combo["description"], str):
            raise ValueError(f"The 'description' value in combination at index {idx} must be a string.")
    
    logger.info("Default metrics YAML structure validated successfully.")
    return defaults

defaults_config = load_default_metrics(DEFAULTS_FILE)

def get_metrics_by_class(evaluate_class: str, infer_class: str) -> list:
    """
    Retrieves the list of metrics configured for a given evaluation and inference class combination.

    Parameters:
        evaluate_class (str): The class name for evaluation purposes.
        infer_class (str): The class name used for inference.

    Returns:
        list: A list of metric names applicable to the given class combination.

    Raises:
        ValueError: If no matching combination is found for the given evaluation and inference classes.
        RuntimeError: If the configuration for combinations is missing or improperly loaded.

    This function iterates over pre-loaded configuration entries to find a match for the provided class names.
    If a matching combination is found, it returns the associated list of metrics. If no such combination exists,
    a ValueError is raised indicating the absence of a valid combination for the provided classes.
    """
    # Check if the global configuration is loaded
    if 'combinations' not in defaults_config:
        raise RuntimeError("Metric combinations configuration is missing or not loaded correctly.")

    # Iterate over the list of combinations in the configuration
    for entry in defaults_config["combinations"]:
        if entry["evaluate"] == evaluate_class and entry["infer"] == infer_class:
            return entry["metrics"]

    # If no combination is found that matches the input parameters, raise an error
    raise ValueError(f"No metrics found for evaluate class '{evaluate_class}' and infer class '{infer_class}'.")
