#!/usr/bin/env python3

import networkx as nx

from sctram.input._input_trajectory import InputTrajectory
from sctram.utils._constants import key_trajectories


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
    
    def get_complete(self, include_additional_nodes):
        trajectory_subgraph = InputTrajectory()
        trajectory_subgraph.graph.update(self.graph)
        trajectory_subgraph.graph[key_trajectories] = "complete"

        for u, v, _, edge_attrs in self.edges(keys=True, data=True):            
            if not trajectory_subgraph.has_edge(u, v):
                filtered_edge_attrs = {k: v for k, v in edge_attrs.items() if k != key_trajectories}
                trajectory_subgraph.add_edge(u, v, **filtered_edge_attrs)
            
        for n, node_attrs in self.nodes(data=True):
            is_additional = node_attrs[key_trajectories] is None
            if not is_additional or include_additional_nodes:
                filtered_node_attrs = {k: v for k, v in node_attrs.items() if k != key_trajectories}
                if n in trajectory_subgraph:
                    trajectory_subgraph.nodes[n].update(filtered_node_attrs)
                else:
                    trajectory_subgraph.add_node(n, **filtered_node_attrs)

        return trajectory_subgraph
        

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
