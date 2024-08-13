#!/usr/bin/env python3

import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Union

import networkx as nx

# Custom Typings
InputGraphDictWithoutEdge = Dict[str, Iterable[tuple[str, str]]]
InputGraphDictNoneEdge = Dict[str, Iterable[tuple[str, str, None]]]
InputGraphDictWithEdge = Dict[str, List[tuple[str, str, dict]]]
InputGraphPossibleTypes = Union[InputGraphDictWithoutEdge, InputGraphDictNoneEdge, InputGraphDictWithEdge]

# Constants
key_trajectories = "trajectories"
edge_reserved_keys = [key_trajectories]
node_reserved_keys = [key_trajectories]

# Logger
logger = logging.getLogger(name="read_trajectories")


class InputTrajectory(nx.DiGraph):
    """_summary_."""

    pass


class InputTrajectories(nx.MultiDiGraph):
    """_summary_."""

    def get_trajectory(self, trajectory: str, include_additional_nodes: bool) -> InputTrajectory:
        """Returns a subgraph containing all edges where the 'key_trajectories' attribute is equal to 'trajectory'.

        Args:
            trajectory (str): The trajectory key to filter the edges by.
            include_additional_nodes(bool): Whether or not include isolated nodes.

        Returns:
            InputTrajectory: A subgraph containing the filtered edges and corresponding nodes,
                preserving all node attributes, edge attributes, and graph-level attributes.
        """
        # Initialize an empty directed graph for the subgraph
        trajectory_subgraph = InputTrajectory()

        # Copy the graph-level attributes
        trajectory_subgraph.graph.update(self.graph)
        trajectory_subgraph.graph[key_trajectories] = trajectory

        # Iterate over all edges in the current graph
        for u, v, _, edge_attrs in self.edges(keys=True, data=True):
            # Check if the 'key_trajectories' attribute of the edge matches the given trajectory.
            if edge_attrs[key_trajectories] == trajectory:
                # Add the edge to the subgraph with its associated attributes
                filtered_edge_attrs = {k: v for k, v in edge_attrs.items() if k != key_trajectories}
                trajectory_subgraph.add_edge(u, v, **filtered_edge_attrs)

        # Iterate over all nodes in the current graph
        for n, node_attrs in self.nodes(data=True):
            # Check if the 'key_trajectories' attribute of the node matches the given trajectory.
            is_additional = node_attrs[key_trajectories] is None
            if (is_additional and include_additional_nodes) or (
                not is_additional and trajectory in node_attrs[key_trajectories]
            ):  # as it is in list format for nodes.
                # Create a new dictionary without the 'key_trajectories' attribute
                filtered_node_attrs = {k: v for k, v in node_attrs.items() if k != key_trajectories}
                if n in trajectory_subgraph:
                    trajectory_subgraph.nodes[n].update(filtered_node_attrs)
                else:
                    trajectory_subgraph.add_node(n, **filtered_node_attrs)

        return trajectory_subgraph

    def _check_individual_trajectories(self):
        """_summary_."""
        for trajectory in self.graph[key_trajectories]:
            trajectory_subgraph = self.get_trajectory(trajectory=trajectory, include_additional_nodes=False)
            if not trajectory_subgraph.is_directed():
                raise ValueError(f"Trajectory {trajectory!r} is not directed.")
            elif trajectory_subgraph.is_multigraph():
                raise ValueError(f"Trajectory {trajectory!r} is a multigraph, which is not allowed.")
            elif nx.number_of_selfloops(trajectory_subgraph) != 0:
                raise ValueError(f"Trajectory {trajectory!r} is contains self-loops.")

    def verify(self):
        """_summary_."""
        self._check_individual_trajectories()
        pass


def convert_to_dict_tuples(obj: InputGraphPossibleTypes) -> InputGraphDictWithEdge:
    """_summary_."""
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
    """_summary_."""
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
    """_summary_."""
    # Check for self-loops
    if start_node == end_node:
        return f"Self-loop detected: Edge from {start_node!r} to itself is not allowed."

    # Check for misuse of reserved keys in edge attributes
    for key in attrs:
        if not isinstance(key, str):
            return f"Key {key!r} for edge attribution {start_node!r} to {end_node!r} for must be a string."

        if key in edge_reserved_keys:
            return (
                f"Reserved key {key!r} cannot be used as an attribute for the edge {start_node!r} to {end_node!r}. "
                f"Reserved keys include: {', '.join(edge_reserved_keys)!r}."
            )

    return None


def is_valid_node_attributes(node: str, attrs: dict[str, Any]) -> Optional[str]:
    """_summary_."""
    # Check for misuse of reserved keys in node attributes
    for key in attrs:
        if not isinstance(key, str):
            return f"Key {key!r} for node attribution {node!r} for must be a string."

        if key in node_reserved_keys:
            return (
                f"Reserved key {key!r} should not be used as an attribute for node {node!r}. "
                f"Reserved keys include: {', '.join(node_reserved_keys)!r}."
            )

    return None


def is_valid_node_attributes_structure(obj: dict[str, dict[str, Any]]) -> Optional[str]:
    """_summary_."""
    if not isinstance(obj, dict):
        return "error message"  # TODO

    for key in obj.keys():
        if not isinstance(key, str):
            return f"Key {key!r} for a node attribute for must be a string."

    return None


def is_valid_additional_nodes(obj: Iterable[str]) -> Optional[str]:
    """_summary_."""
    if len({i for i in obj}) != len([i for i in obj]):
        return "Additional nodes should be an iterable with unique elements."

    return None


def raise_error(response):
    """_summary_."""
    if response is not None:
        raise ValueError(response)


def read(
    ground_truth_trajectories: InputGraphPossibleTypes,
    additional_nodes: Optional[Iterable[str]] = None,
    node_attributes: Optional[dict[str, dict[str, Any]]] = None,
) -> InputTrajectories:
    """_summary_."""
    # Validate input structure, and convert tuples to have dict as third element if missing
    raise_error(is_valid_input_structure(obj=ground_truth_trajectories))
    ground_truth_trajectories = convert_to_dict_tuples(obj=ground_truth_trajectories)

    # Initialize the graph
    gtt = InputTrajectories()

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
            logger.warning(f"Edge ({u}, {v}) is missing attribute keys: {missing_keys}")
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

    # If 'addditional_nodes' are provided, integrate it into the graph
    if additional_nodes is not None:
        raise_error(is_valid_additional_nodes(obj=additional_nodes))
        for node in additional_nodes:
            if node in gtt.nodes:
                raise ValueError(f"Additional node {node} already exist in the graph.")
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
                logger.warning(f"Node {node} is missing attribute keys: {missing_keys}")
                for key in missing_keys:
                    gtt.nodes[node][key] = None

    gtt.verify()
    return gtt
