#!/usr/bin/env python3

import glob
import numpy as np
import os 
from loguru import logger
import time
import yaml
import importlib
import importlib.util
from functools import partial
from typing import Optional


_logger = logger.bind(name="MetricConfig")


class MetricConfigError(Exception):
    """Custom exception for metric configuration errors."""
    pass


def metric_decorator(category: Optional[str] = None, message: Optional[str] = None):
    def decorator(func):
        def wrapper(*args, return_description: bool, **kwargs):
            
            error_message = None            
            start_time = time.perf_counter()  # Start timing

            # Calculate the metric
            try:
                score = func(*args, **kwargs)
            except Exception as e:
                score = np.nan
                error_message = f"{str(e)} (Line {e.__traceback__.tb_lineno})"

            elapsed_time = time.perf_counter() - start_time  # Stop timing

            # Check if the description needs to be returned based on runtime argument
            if return_description:
                description_string = description_creator(
                    message=message,
                    category=category,
                    score=score,
                    time=elapsed_time,
                    error_message=error_message
                )
                return (score, description_string)
            else:
                return score

        return wrapper
    return decorator


def description_creator(message: str, category: str, score: float, time: float, error_message: Optional[str] = None) -> str:
    """Creates a concise, single-line description string for logging purposes, emphasizing clarity, precision.

    Parameters:
        message (str): A detailed description of the metric.
        category (str): The category to which the metric belongs. Must be one of 'adjacency', 'pseudotime', or 'embedding'.
        score (float): The computed metric score.
        time (float): The execution time in seconds, formatted to appear last in the description.
        error_message (Optional[str]): The error message and its line number if an exception occurred.

    Returns:
        str: A professional single-line formatted string summarizing the metric, ideal for logging.

    Raises:
        MetricConfigError: If an unsupported category is provided.
    """
    # Define a dictionary to convert category codes to human-readable strings.
    category_pretty_dict = {
        "adjacency": "Adjacency-based",
        "pseudotime": "Pseudotime-based",
        "embedding": "Embedding-based"
    }
    
    if message is None or category is None:
        raise MetricConfigError("Both 'message' and 'category' must be provided in metric decarator, and cannot be None.")

    # Attempt to retrieve the pretty format for the category, raising a clear error if the category is not supported.
    try:
        category_pretty = category_pretty_dict[category]
    except KeyError:
        raise MetricConfigError(f"Unsupported category '{category}'. Valid categories are: {', '.join(category_pretty_dict.keys())}.")

    # Convert the score to string and format the time with precision.
    score_str = str(score)
    time_str = f"{time:.4f} sec"
    
    # Compose the description string
    description = f"Metric: {message!r} ({category_pretty}), Score: {score_str!r}, Computation Time: {time_str!r}"
    if error_message:
        description += f", Error: {error_message!r}"
        
    return description


def load_and_validate_metrics_config(config_path):
    """Validate the structure and contents of the metrics configuration."""
    try:  # Load and parse the YAML file.
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise MetricConfigError(f"Failed to read or parse the configuration file '{config_path}': {e}")
    
    if not isinstance(config, dict):
        raise MetricConfigError("The configuration must be a dictionary.")

    if "metrics" not in config:
        raise MetricConfigError("The configuration is missing the 'metrics' key.")

    if not isinstance(config["metrics"], list):
        raise MetricConfigError("The 'metrics' key must be associated with a list.")

    directory = os.path.dirname(config_path)  # get the directory of the YAML file
    referenced_pairs = set()
    referenced_base = set()

    for idx, metric_def in enumerate(config["metrics"], start=1):
        required_keys = {
            "base_function": str,
            "module": str,
            "category": str,
            "message": str,
            "validate_result": bool
        }

        for key, expected_type in required_keys.items():
            if key not in metric_def:
                raise MetricConfigError(f"Metric definition #{idx} is missing the '{key}' key.")
            if not isinstance(metric_def[key], expected_type):
                raise MetricConfigError(f"Metric definition #{idx} has an incorrect type for key '{key}'.")

        module_path = f"{__package__}.{metric_def['module']}" if __package__ else metric_def["module"]
        if importlib.util.find_spec(module_path) is None:
            raise MetricConfigError(f"Module '{module_path}' does not exist.")
        
        pair = (metric_def["module"], metric_def["base_function"])
        if pair in referenced_pairs:
            raise MetricConfigError(f"Function {metric_def['base_function']!r} in Module {module_path!r} configured more than one times.")
        else:
            referenced_pairs.add(pair)
            
        if metric_def["base_function"] in referenced_base:
            raise MetricConfigError(f"Function {metric_def['base_function']!r} configured more than one times.")
        else:
            referenced_base.add(metric_def["base_function"])
        
    all_python_modules = set(
        os.path.splitext(os.path.basename(f))[0] 
        for f in glob.glob(f"{directory}/_*.py") 
        if not f.endswith('__init__.py')
    )
    referenced_modules = {i for i, _ in referenced_pairs}
    unused_modules = all_python_modules - referenced_modules
    if unused_modules:
        raise MetricConfigError(f"Module is missing in config: {', '.join(unused_modules)}.")

    return config


def load_metrics(config_path: str):
    config = load_and_validate_metrics_config(config_path=config_path)
    metrics = {}
    # Use __package__ to build the module path relative to the current package.
    base_repo = __package__  # e.g., "sctram.evaluate.metrics"
    for metric_def in config["metrics"]:
        # Build the full module path. If __package__ is empty, use the module name as is.
        module_path = f"{base_repo}.{metric_def['module']}" if base_repo else metric_def["module"]
        # Dynamically import the module.
        module = importlib.import_module(module_path)
        # Retrieve the base metric function.
        base_func_before_validation = getattr(module, metric_def["base_function"])
        # Enter the validation if needed.
        base_func = partial(base_func_before_validation, validate_result=metric_def["validate_result"])

        # Decorate the function.
        decorated_func = metric_decorator(
            category=metric_def["category"],
            message=metric_def["message"]
        )(base_func)

        # Create the version with the return_description parameter set to True.
        metric_with_desc = partial(decorated_func, return_description=True)
        metric_without_desc = partial(decorated_func, return_description=False)

        # Store in a dict with the metric's name as key.
        metrics[metric_def["base_function"]] = {
            "base_before_val": base_func_before_validation,
            "base": base_func,
            "without_desc": metric_without_desc,
            "with_desc": metric_with_desc
        }

    return metrics

# Get the absolute path to the directory where metric config located
base_directory = os.path.dirname(os.path.abspath(__file__))
absolute_path_metrics_yaml = os.path.join(base_directory, "metrics.yaml")

metrics = load_metrics(absolute_path_metrics_yaml)
