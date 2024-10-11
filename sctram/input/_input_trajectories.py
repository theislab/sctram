#!/usr/bin/env python3

import networkx as nx

from sctram.input._input_trajectory import InputTrajectory
from sctram._constants import key_trajectories


class InputTrajectories(nx.MultiDiGraph):
    """_summary_."""  # TODO

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
        """_summary_."""  # TODO
        for trajectory in self.graph[key_trajectories]:
            trajectory_subgraph = self.get_trajectory(trajectory=trajectory, include_additional_nodes=False)
            if not trajectory_subgraph.is_directed():
                raise ValueError(f"Trajectory {trajectory!r} is not directed.")
            elif trajectory_subgraph.is_multigraph():
                raise ValueError(f"Trajectory {trajectory!r} is a multigraph, which is not allowed.")
            elif nx.number_of_selfloops(trajectory_subgraph) != 0:
                raise ValueError(f"Trajectory {trajectory!r} is contains self-loops.")

    def verify(self):
        """_summary_."""  # TODO
        self._check_individual_trajectories()
