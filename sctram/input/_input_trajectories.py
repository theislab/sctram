#!/usr/bin/env python3

import networkx as nx
import pickle
from typing import Optional, Iterable, Any, Union

from sctram.utils._constants import InputGraphPossibleTypes, key_trajectories
from sctram.input._input_trajectory import InputTrajectory
from sctram.input._read_trajectories import read_dict


class InputTrajectories(nx.MultiDiGraph):
    """Represents a collection of input trajectories as a multigraph.

    This class extends the NetworkX MultiDiGraph to manage multiple directed trajectories
    within a single graph structure. Each trajectory is identified by a unique key, allowing
    for the isolation and manipulation of specific trajectory subgraphs. This is particularly
    useful in scenarios where multiple pathways or processes coexist within the same network,
    such as in transportation systems, biological pathways, or data flow diagrams.

    By leveraging graph theory principles, this class provides methods to extract, verify,
    and manipulate subgraphs corresponding to individual trajectories, ensuring data integrity
    and consistency across the entire graph structure.

    Attributes:
        key_trajectories (str): The key used to store trajectory identifiers in graph attributes.
    """

    def __init__(
        self,
        ground_truth_trajectories: Optional[InputGraphPossibleTypes] = None,
        additional_nodes: Optional[Iterable[str]] = None,
        node_attributes: Optional[dict[str, dict[str, Any]]] = None,
    ):
        # Initialize parent with all potential NetworkX arguments
        super().__init__()
        
        # Initialize mandatory attributes
        self.graph.setdefault(key_trajectories, [])
        
        # Process trajectories only if provided
        if ground_truth_trajectories is not None:
            new_gtt = read_dict(
                ground_truth_trajectories=ground_truth_trajectories,
                additional_nodes=additional_nodes,
                node_attributes=node_attributes
            )
            self.update_trajectory(new_gtt)

    def get_trajectory(self, trajectory: str, include_additional_nodes: bool) -> InputTrajectory:
        """Extracts a subgraph corresponding to a specific trajectory.

        This method filters the edges of the graph to include only those that belong to the
        specified trajectory, identified by the 'key_trajectories' attribute. It optionally
        includes additional nodes that may be isolated from the trajectory edges, depending on
        the 'include_additional_nodes' flag.

        The resulting subgraph preserves all relevant node and edge attributes, as well as
        graph-level attributes, ensuring that the extracted trajectory maintains its original
        context and properties.

        Args:
            trajectory (str): The identifier for the trajectory to be extracted.
            include_additional_nodes (bool): Determines whether nodes not connected by the
                trajectory's edges should be included in the subgraph. This is useful for
                maintaining the integrity of node attributes or for including ancillary
                information related to the trajectory.

        Returns:
            InputTrajectory: A directed graph representing the specified trajectory, containing
                only the edges and nodes pertinent to that trajectory. This subgraph can be used
                for further analysis, visualization, or processing specific to the trajectory.
        """
        if trajectory not in self.graph[key_trajectories]:
            raise ValueError(f"Trajectory is not found: {trajectory!r}.")
        
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
    
    def get_complete_no_attributes_for_visualization(self):
        trajectory_name = "complete"

        trajectory_subgraph = InputTrajectory()
        trajectory_subgraph.graph[key_trajectories] = trajectory_name

        for u, v, _, _ in self.edges(keys=True, data=True):            
            if not trajectory_subgraph.has_edge(u, v):
                edge_attrs = {key_trajectories: trajectory_name}
                trajectory_subgraph.add_edge(u, v, **edge_attrs)
            
        for n, attrs in self.nodes(data=True):
            if attrs[key_trajectories] is None:
                node_attrs = {key_trajectories: None}
            else:
                node_attrs = {key_trajectories: trajectory_name}
            if n in trajectory_subgraph:
                trajectory_subgraph.nodes[n].update(node_attrs)
            else:
                trajectory_subgraph.add_node(n, **node_attrs)

        return trajectory_subgraph
    
    def update_trajectory(self, new_gtt: Union[nx.MultiDiGraph, "InputTrajectories"]):
        """Merges a new graph into the current instance, updating trajectories and attributes.

        Args:
            new_gtt: The new graph to merge, which must not share any trajectory keys with the current graph.

        Raises:
            ValueError: If there are overlapping trajectory keys or attribute conflicts in nodes.
        """
        if key_trajectories in self.graph:  # The method is updating
            existing_trajs = set(self.graph[key_trajectories])
        elif not (len(self.edges()) == len(self.nodes()) == 0):
            raise ValueError(f"Unexpected error: {self.edges()}, {self.nodes()}")
        else:  # The method is creating
            existing_trajs = set()
            
        new_trajs = set(new_gtt.graph[key_trajectories])
        overlap = existing_trajs & new_trajs
        if overlap:
            raise ValueError(f"Trajectory keys {overlap} already exist in the current graph.")
        
        # Add edges from the new graph
        for u, v, data in new_gtt.edges(data=True):
            self.add_edge(u, v, **data)

        # Merge nodes and their attributes
        for node in new_gtt.nodes():
            new_data = new_gtt.nodes[node]
            if node not in self.nodes:
                self.add_node(node, **new_data)
            else:
                existing_data = self.nodes[node]
                existing_traj = existing_data.get(key_trajectories)
                new_traj = new_data.get(key_trajectories)

                if existing_traj is None:
                    existing_data[key_trajectories] = new_traj.copy() if isinstance(new_traj, list) else new_traj
                elif new_traj is None:
                    pass  # Keep existing trajectories
                else:
                    combined = sorted(set(existing_traj) | set(new_traj))
                    existing_data[key_trajectories] = combined
                
                # Check non-trajectory attributes for consistency
                existing_attrs = {k: v for k, v in existing_data.items() if k != key_trajectories}
                new_attrs = {k: v for k, v in new_data.items() if k != key_trajectories}
                if existing_attrs != new_attrs:
                    raise ValueError(
                        f"Node {node} has conflicting attrs. Existing: {existing_attrs}, New: {new_attrs}"
                    )
                
        # Update the graph's trajectory list
        self.graph[key_trajectories] = sorted(existing_trajs.union(new_trajs))
        # Update other graph attributes from the new graph
        self.graph.update({k: j for k, j in new_gtt.graph.items() if k != key_trajectories})

        # Verify the integrity of the merged graph
        self.verify()
    
    def _check_individual_trajectories(self):
        """Validates the structural integrity of each individual trajectory within the graph.

        This internal method iterates over all trajectory identifiers stored in the graph-level
        'key_trajectories' attribute and performs a series of checks to ensure that each trajectory
        adheres to the required graph properties. Specifically, it verifies that:

        - Each trajectory subgraph is directed, ensuring the directional flow of the trajectory.
        - Each trajectory subgraph is not a multigraph, preventing multiple parallel edges which
          could lead to ambiguity in trajectory representation.
        - Each trajectory subgraph does not contain self-loops, maintaining the acyclic nature of
          the trajectory and avoiding trivial or redundant cycles.

        These checks are essential to maintain the correctness and usability of the trajectory data,
        ensuring that downstream processes can operate on well-defined and reliable graph structures.

        Raises:
            ValueError: If any trajectory fails to meet the directed, non-multigraph, or
                acyclic (no self-loops) criteria, an error is raised with a descriptive message.
        """
        for trajectory in self.graph[key_trajectories]:
            trajectory_subgraph = self.get_trajectory(trajectory=trajectory, include_additional_nodes=False)
            if not trajectory_subgraph.is_directed():
                raise ValueError(f"Trajectory {trajectory!r} is not directed.")
            elif trajectory_subgraph.is_multigraph():
                raise ValueError(f"Trajectory {trajectory!r} is a multigraph, which is not allowed.")
            elif nx.number_of_selfloops(trajectory_subgraph) != 0:
                raise ValueError(f"Trajectory {trajectory!r} contains self-loops.")
            elif not nx.is_weakly_connected(trajectory_subgraph):
                raise ValueError(f"Trajectory {trajectory!r} is not weakly connected.")

    def verify(self):
        """Performs comprehensive verification of all trajectories within the graph.

        This method serves as an external interface to validate the structural integrity of each
        trajectory contained within the graph. It leverages the internal '_check_individual_trajectories'
        method to ensure that all trajectories conform to the necessary graph properties, including
        being directed, non-multigraph, and free of self-loops.

        This verification step is crucial before any analysis or processing is performed on the
        trajectories, guaranteeing that the data adheres to the expected standards and preventing
        potential errors or inconsistencies in downstream operations.
        """
        self._check_individual_trajectories()
        
    # def save(self, path: str) -> None:
    #     """Serialize the InputTrajectories instance to a file using pickle.
        
    #     Args:
    #         path: Path to the output file.
    #     """
    #     with open(path, 'wb') as f:
    #         pickle.dump(self, f)

    # @classmethod
    # def load(cls, path: str) -> "InputTrajectories":
    #     """Safe loading method with exact type checking"""
    #     with open(path, "rb") as f:
    #         obj = pickle.load(f)
    #     if type(obj) is not cls:
    #         raise TypeError(f"Loaded object is type {type(obj)}, expected {cls}")
    #     return obj
    
    def __repr__(self):
        return (
            f"InputTrajectories(trajectories={len(self.graph[key_trajectories])}, "
            f"nodes={self.number_of_nodes()}, edges={self.number_of_edges()})"
        )
    