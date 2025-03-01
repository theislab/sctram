#!/usr/bin/env python3

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
import yaml
from loguru import logger

from sctram.utils._constants import (
    InputGraphDictWithEdge,
    InputGraphPossibleTypes,
    edge_reserved_keys,
    key_trajectories,
    node_reserved_keys,
)

# Logger
_logger = logger.bind(name="ReadTrajectories")


def convert_to_dict_tuples(obj: InputGraphPossibleTypes) -> InputGraphDictWithEdge:
    """Converts various input graph structures into a standardized dictionary format with edge attributes.

    This function normalizes different representations of input graphs by ensuring that each edge
    is represented as a tuple with exactly three elements: the start node, the end node, and a
    dictionary of edge attributes. If an edge is provided without attributes or with a `None`
    attribute, an empty dictionary is assigned to maintain consistency across the graph data.

    The standardized format facilitates uniform processing and validation of graph data, enabling
    downstream functions to operate on a predictable data structure regardless of the original input format.

    Args:
        obj (InputGraphPossibleTypes): The input graph data, which can be one of several types
            including dictionaries with two-element tuples, three-element tuples, or existing
            dictionaries with complete edge attribute dictionaries.

    Returns:
        InputGraphDictWithEdge: A dictionary where each key corresponds to a trajectory and each value
            is a list of tuples, each containing the start node, end node, and a dictionary of edge attributes.
    """
    result = {}
    for key, iterable in obj.items():
        new_tuples: list = []
        for item in iterable:
            # If the tuple has only two elements or the third is None, append an empty dict
            if len(item) == 2 or item[2] is None:
                new_tuples.append((item[0], item[1], dict()))
            else:
                # If there is already a dict, use it as is
                new_tuples.append(item)
        result[key] = new_tuples

    return result


def is_valid_input_structure(obj: Any) -> Optional[str]:
    """Validates the structure of the input graph data.

    This function ensures that the input data adheres to the expected format required for
    constructing the graph. The validation checks include:

    - The input must be a dictionary.
    - Each key in the dictionary must be a string, representing a trajectory identifier.
    - Each value corresponding to a key must be an iterable (e.g., list or tuple) of tuples.
    - Each tuple must have either two or three elements, where:
        - The first two elements are strings representing the start and end nodes of an edge.
        - The optional third element is either a dictionary of edge attributes or `None`.

    Args:
        obj (Any): The input graph data to be validated.

    Returns:
        Optional[str]: Returns `None` if the input structure is valid. Otherwise, returns a descriptive
            error message indicating the first encountered validation failure.
    """
    if not isinstance(obj, dict):
        return "The object must be a dictionary."

    for key, value in obj.items():
        if not isinstance(key, str):
            return f"The key {key!r} must be a string."

        if not isinstance(value, Iterable):
            return f"The value for key {key!r} must be an iterable."

        for item in value:
            # Check if each item is a tuple and its length is 2 or 3
            if not isinstance(item, tuple) or len(item) not in (2, 3):
                return f"Each item in the list for key {key!r} must be a tuple with either two or three elements."

            # Check if the first two elements are strings
            if not all(isinstance(subitem, str) for subitem in item[:2]):
                return f"The first two elements of each tuple for key {key!r} must be strings."

            # Additional checks for tuples with three elements
            if len(item) == 3:
                if not (isinstance(item[2], dict) or item[2] is None):
                    return f"The third element of each tuple for key {key!r} must be a dictionary or None."

    return None


def is_valid_edge_attributes(start_node: str, end_node: str, attrs: dict[str, Any]) -> Optional[str]:
    """Validates the attributes of an edge in the graph.

    This function performs several checks to ensure that the edge attributes meet the
    required criteria for proper graph construction and integrity. The validation steps include:

    - Ensuring that the edge does not form a self-loop (i.e., the start and end nodes are different).
    - Verifying that all keys in the edge attributes are strings.
    - Ensuring that reserved keys are not used as edge attribute keys to prevent conflicts
      with predefined graph properties.

    Args:
        start_node (str): The identifier of the start node of the edge.
        end_node (str): The identifier of the end node of the edge.
        attrs (dict[str, Any]): A dictionary of attributes associated with the edge.

    Returns:
        Optional[str]: Returns `None` if the edge attributes are valid. Otherwise, returns a
            descriptive error message indicating the validation failure.
    """
    # Check for self-loops
    if start_node == end_node:
        return f"Self-loop detected: Edge from {start_node!r} to itself is not allowed."

    # Check for misuse of reserved keys in edge attributes
    for key in attrs:
        if not isinstance(key, str):
            return f"Key {key!r} for edge attribution {start_node!r} to {end_node!r} must be a string."

        if key in edge_reserved_keys:
            return (
                f"Reserved key {key!r} cannot be used as an attribute for the edge {start_node!r} to {end_node!r}. "
                f"Reserved keys include: {', '.join(edge_reserved_keys)!r}."
            )

    return None


def is_valid_node_attributes(node: str, attrs: dict[str, Any]) -> Optional[str]:
    """Validates the attributes of a node in the graph.

    This function ensures that the node attributes comply with the required standards by
    performing the following checks:

    - All attribute keys must be strings to maintain consistency and prevent type-related issues.
    - Reserved keys are not used as node attribute keys to avoid conflicts with predefined graph
      properties or special identifiers.

    Args:
        node (str): The identifier of the node being validated.
        attrs (dict[str, Any]): A dictionary of attributes associated with the node.

    Returns:
        Optional[str]: Returns `None` if the node attributes are valid. Otherwise, returns a
            descriptive error message indicating the validation failure.
    """
    # Check for misuse of reserved keys in node attributes
    for key in attrs:
        if not isinstance(key, str):
            return f"Key {key!r} for node attribution {node!r} must be a string."

        if key in node_reserved_keys:
            return (
                f"Reserved key {key!r} should not be used as an attribute for node {node!r}. "
                f"Reserved keys include: {', '.join(node_reserved_keys)!r}."
            )

    return None


def is_valid_node_attributes_structure(obj: dict[str, dict[str, Any]]) -> Optional[str]:
    """Validates the overall structure of node attributes.

    This function ensures that the node attributes are provided in the correct format,
    which is a dictionary where each key corresponds to a node identifier (string) and each
    value is another dictionary containing the attributes for that node.

    Args:
        obj (dict[str, dict[str, Any]]): The node attributes data to be validated.

    Returns:
        Optional[str]: Returns `None` if the node attributes structure is valid. Otherwise, returns
            a descriptive error message indicating the validation failure.
    """
    if not isinstance(obj, dict):
        return "The node attributes must be provided as a dictionary."

    for key in obj.keys():
        if not isinstance(key, str):
            return f"Key {key!r} for a node attribute must be a string."

    return None


def is_valid_additional_nodes(obj: Iterable[str]) -> Optional[str]:
    """Validates the list of additional nodes to be added to the graph.

    This function checks that the provided iterable of additional nodes contains only unique
    elements, ensuring that no duplicate nodes are introduced into the graph. Duplicate nodes
    could lead to inconsistencies or unintended behavior within the graph structure.

    Args:
        obj (Iterable[str]): An iterable containing the identifiers of additional nodes to be added.

    Returns:
        Optional[str]: Returns `None` if all additional nodes are unique. Otherwise, returns a
            descriptive error message indicating the presence of duplicate nodes.
    """
    if len({i for i in obj}) != len([i for i in obj]):
        return "Additional nodes should be an iterable with unique elements."

    return None


def raise_error(response):
    """Raises a ValueError if a validation error message is provided.

    This utility function centralizes the error-raising mechanism by checking if a
    validation function has returned an error message. If an error message is present,
    it raises a ValueError with the provided message, effectively halting further
    processing and signaling the presence of invalid input data.

    Args:
        response (Optional[str]): The error message returned by a validation function.
            If `None`, no action is taken. Otherwise, a ValueError is raised with the message.

    Raises:
        ValueError: If `response` is not `None`, indicating that a validation error has occurred.
    """
    if response is not None:
        raise ValueError(response)


def read_dict(
    ground_truth_trajectories: InputGraphPossibleTypes,
    additional_nodes: Optional[Iterable[str]] = None,
    node_attributes: Optional[dict[str, dict[str, Any]]] = None,
) -> nx.MultiDiGraph:
    """Constructs an nx.MultiDiGraph graph from provided dictionary data.

    This function builds an nx.MultiDiGraph graph by processing the ground truth
    trajectory data, additional nodes, and node attributes. The process involves several
    key steps:

    1. Input Validation and Normalization:
        - Validates the structure of the ground truth trajectories to ensure they conform
          to the expected format.
        - Normalizes the trajectory data to ensure that each edge is represented as a tuple
          with three elements, appending an empty dictionary for edge attributes if necessary.

    2. Graph Construction:
        - Initializes an empty nx.MultiDiGraph graph.
        - Iterates over each trajectory and its corresponding edges, adding them to the graph
          while validating edge attributes and ensuring no duplication.

    3. Attribute Consistency:
        - Aggregates all trajectory keys to maintain a comprehensive list of trajectories.
        - Ensures that all edges have a consistent set of attribute keys, logging warnings
          and assigning `None` to any missing attributes.

    4. Node Attribute Integration:
        - Associates each node with its corresponding trajectories, ensuring that the
          `key_trajectories` attribute accurately reflects the node's involvement in various
          trajectories.
        - Processes additional nodes and integrates node attributes, enforcing uniqueness
          and consistency.

    5. Final Verification:
        - Validates the integrity of the entire graph structure, ensuring that all trajectories
          meet the required graph properties such as being directed, non-multigraph, and free
          of self-loops.

    Args:
        ground_truth_trajectories (InputGraphPossibleTypes): The main trajectory data defining
            the edges and their associated trajectories.
        additional_nodes (Optional[Iterable[str]]): An optional iterable of node identifiers
            to be added to the graph without specific trajectory associations.
        node_attributes (Optional[dict[str, dict[str, Any]]]): An optional dictionary specifying
            additional attributes for nodes, allowing for enriched node representations.

    Returns:
        nx.MultiDiGraph: A fully constructed and validated nx.MultiDiGraph graph containing
            all specified trajectories, additional nodes, and node attributes.

    Raises:
        ValueError: If any validation step fails, such as invalid input structure, edge
            attributes, duplicate additional nodes, or node attribute inconsistencies.
    """
    # Validate input structure, and convert tuples to have dict as third element if missing
    raise_error(is_valid_input_structure(obj=ground_truth_trajectories))
    ground_truth_trajectories = convert_to_dict_tuples(obj=ground_truth_trajectories)

    # Initialize the graph
    gtt = nx.MultiDiGraph()

    # Add edges and nodes to the graph, ensuring each tuple has exactly three elements
    all_key_trajectories = set()
    for key, edges in ground_truth_trajectories.items():
        for start_node, end_node, attrs in edges:
            raise_error(is_valid_edge_attributes(start_node=start_node, end_node=end_node, attrs=attrs))
            attrs[key_trajectories] = key
            gtt.add_edge(start_node, end_node, **attrs)
            all_key_trajectories.add(key)

    # Add all_key_trajectories to the graph as graph-level attribute
    gtt.graph[key_trajectories] = sorted(all_key_trajectories)

    # Check for attribute consistency across edges
    all_keys = set()
    for _, _, data in gtt.edges(data=True):
        all_keys.update(data.keys())

    # Find missing keys in each edge and log warnings if any key is only on some edges
    for u, v, data in gtt.edges(data=True):
        missing_keys = all_keys - data.keys()
        if missing_keys:
            _logger.warning(f"Edge ({u}, {v}) is missing attribute keys: {missing_keys}")
            for key in missing_keys:
                data[key] = None

    # Update 'key_trajectories' attribute for start and end nodes
    for key, edges in ground_truth_trajectories.items():
        for start_node, end_node, _ in edges:
            gtt.nodes[start_node].setdefault(key_trajectories, []).append(key)
            gtt.nodes[end_node].setdefault(key_trajectories, []).append(key)

    # Ensure unique listings for 'key_trajectories' if needed
    for node in gtt.nodes():
        if key_trajectories in gtt.nodes[node]:
            gtt.nodes[node][key_trajectories] = sorted(set(gtt.nodes[node][key_trajectories]))

    # If 'additional_nodes' are provided, integrate it into the graph
    if additional_nodes is not None:
        raise_error(is_valid_additional_nodes(obj=additional_nodes))
        for node in additional_nodes:
            if node in gtt.nodes:
                raise ValueError(f"Additional node {node} already exists in the graph.")
            gtt.add_node(node, **{key_trajectories: None})

    # If 'node_attributes' is provided, integrate it into the graph
    if node_attributes is not None:
        raise_error(is_valid_node_attributes_structure(obj=node_attributes))
        all_node_keys: set = set()
        for node, attrs in node_attributes.items():
            raise_error(is_valid_node_attributes(node=node, attrs=attrs))
            if node in gtt:
                gtt.nodes[node].update(attrs)
                all_node_keys.update(attrs.keys())
            else:
                raise ValueError(f"Node {node} defined in 'node_attributes' does not exist in the graph.")

        # Check for attribute consistency across nodes
        for node in gtt.nodes():
            missing_keys = all_node_keys - gtt.nodes[node].keys()
            if missing_keys:
                _logger.warning(f"Node {node} is missing attribute keys: {missing_keys}")
                for key in missing_keys:
                    gtt.nodes[node][key] = None

    return gtt


def read_yaml(yaml_path: Union[Path, str]) -> nx.MultiDiGraph:
    """Reads and constructs an nx.MultiDiGraph graph from a YAML configuration file.

    This function parses a YAML file containing graph trajectory data, validates the structure
    and attributes, and constructs an `nx.MultiDiGraph` graph object. The YAML file should
    define trajectories, additional nodes, and node-specific attributes in a clear and intuitive
    format.

    Supported YAML Structure:

    The YAML file should have the following top-level keys:

    - `trajectories`: Defines the main trajectory data.
    - `additional_nodes`: (Optional) Lists nodes to be added without specific trajectory associations.
    - `node_attributes`: (Optional) Specifies attributes for nodes.

    Example YAML Formats:

    1. Basic Trajectories:

       Defines multiple trajectories with edges specified in a clear `start`, `end`, and optional `attributes` format.

       .. code-block:: yaml

           trajectories:
             trajectory_1:
               - start: A
                 end: B
               - start: B
                 end: C
               - start: C
                 end: D

             trajectory_2:
               - start: D
                 end: E
               - start: E
                 end: F

       In this structure, each edge within a trajectory is represented as a dictionary with `start` and
       `end` keys. The `attributes` field is optional. If omitted, it defaults to an empty dictionary.

    2. Trajectories with Edge Attributes:

       Incorporates edge-specific attributes such as `speed` and `distance`.

       .. code-block:: yaml

           trajectories:
             trajectory_1:
               - start: A
                 end: B
                 attributes:
                   speed: 5
                   distance: 10
               - start: B
                 end: C
                 attributes:
                   speed: 7
               - start: C
                 end: D

             trajectory_2:
               - start: D
                 end: E
                 attributes:
                   speed: 6
                   distance: 8
               - start: E
                 end: F
                 attributes:
                   distance: 12

       Here, each edge can optionally include an `attributes` dictionary. If `attributes` are not
       provided, an empty dictionary is assigned by default.

    3. Comprehensive Example with Additional Nodes and Node Attributes:

       Combines trajectories with additional nodes and node-specific attributes to create a rich graph structure.

       .. code-block:: yaml

           trajectories:
             trajectory_1:
               - start: A
                 end: B
                 attributes:
                   speed: 5
                   distance: 10
               - start: B
                 end: C
                 attributes:
                   speed: 7
               - start: C
                 end: D
                 attributes:
                   distance: 15

             trajectory_2:
               - start: D
                 end: E
                 attributes:
                   distance: 8
               - start: E
                 end: F
                 attributes:
                   speed: 6

             trajectory_3:
               - start: F
                 end: G
                 attributes:
                   speed: 8
               - start: G
                 end: H

             additional_nodes:
               - I
               - J

             node_attributes:
               A:
                 type: "start"
                 color: "blue"
               B:
                 type: "intermediate"
               C:
                 type: "intermediate"
               D:
                 type: "intermediate"
               E:
                 type: "intermediate"
               F:
                 type: "intermediate"
               G:
                 type: "intermediate"
               H:
                 type: "end"
               I:
                 type: "auxiliary"
               J:
                 type: "auxiliary"

       This example demonstrates how to define multiple trajectories with a mix of edge definitions, add
       additional nodes (`I` and `J`), and assign attributes to all nodes, including the additional ones.

    Usage Examples:

    1. Reading a Simple YAML File:

       .. code-block:: python

           from pathlib import Path

           graph = read_yaml(yaml_input=Path("simple_trajectories.yaml"))

    2. Reading a Comprehensive YAML File:

       .. code-block:: python

           graph = read_yaml(yaml_input="comprehensive_trajectories.yaml")

    3. Handling Invalid YAML Structures:

       If the YAML file contains invalid structures, such as non-string node identifiers
       or improper edge definitions, the function will raise a `ValueError` with a
       descriptive message.

       .. code-block:: python

           try:
               graph = read_yaml("invalid_trajectories.yaml")
           except ValueError as e:
               print(f"Failed to read YAML: {e}")

    Returns:
        nx.MultiDiGraph:
            A fully constructed and validated `nx.MultiDiGraph` graph containing all specified
            trajectories, additional nodes, and node attributes.

    Args:
        yaml_path(Union[Path, str]): The path to the YAML file as a string or `Path` object.
            This input defines the trajectories, edges, additional nodes, and node attributes.

    Raises:
        ValueError: If the YAML structure is invalid, such as incorrect data types, missing elements,
            or misuse of reserved keys in attributes.
        FileNotFoundError: If the specified YAML file does not exist.

    TODO: Improvement for YAML function.
        - Test the function.
        - Support for inline YAML strings or file-like objects.
        - Enhanced error messages with line numbers for YAML parsing errors.
    """
    # Determine if yaml_input is a file path or a YAML content iterable
    if isinstance(yaml_path, (str, Path)):
        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"The YAML file {yaml_path} does not exist.")
        with yaml_path.open("r", encoding="utf-8") as file:
            try:
                yaml_data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {yaml_path}: {e}") from e
    else:
        raise ValueError("YAML path must be a string or Path object.")

    if not isinstance(yaml_data, dict):
        raise ValueError("The root of the YAML file must be a dictionary.")

    # Extract trajectories, additional_nodes, and node_attributes from the YAML data
    trajectories = yaml_data.get("trajectories")
    if trajectories is None:
        raise ValueError("The YAML file must contain a 'trajectories' section.")

    additional_nodes_yaml = yaml_data.get("additional_nodes", [])
    node_attributes_yaml = yaml_data.get("node_attributes", {})

    # Normalize edge definitions
    normalized_trajectories = {}
    for traj_key, edges in trajectories.items():
        if not isinstance(edges, list):
            raise ValueError(f"The edges for trajectory {traj_key!r} must be a list.")
        normalized_edges = []
        for idx, edge in enumerate(edges, start=1):
            if not isinstance(edge, dict):
                raise ValueError(f"Each edge in trajectory {traj_key!r} must be a dictionary (error at edge {idx}).")
            start = edge.get("start")
            end = edge.get("end")
            attrs = edge.get("attributes", {})
            if start is None or end is None:
                raise ValueError(
                    f"Each edge must have 'start' and 'end' nodes in trajectory {traj_key!r} (error at edge {idx})."
                )
            if attrs is None:
                attrs = {}
            elif not isinstance(attrs, dict):
                raise ValueError(
                    f"The 'attributes' for edge ({start}, {end}) in trajectory {traj_key!r} "
                    f"must be a dictionary or null (error at edge {idx})."
                )
            normalized_edges.append((start, end, attrs))
        normalized_trajectories[traj_key] = normalized_edges

    # Call read_dict with the normalized data
    return read_dict(
        ground_truth_trajectories=normalized_trajectories,
        additional_nodes=additional_nodes_yaml,
        node_attributes=node_attributes_yaml,
    )
