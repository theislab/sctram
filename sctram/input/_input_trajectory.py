#!/usr/bin/env python3

from functools import cached_property
from typing import Any, Dict, FrozenSet, List, Set

import networkx as nx
from loguru import logger

from sctram.utils._constants import key_trajectories

_logger = logger.bind(name="InputTrajectory")


class InputTrajectory(nx.DiGraph):
    """Represents a single input trajectory as a directed graph.

    This class extends the NetworkX DiGraph to encapsulate a single trajectory within the
    broader collection of trajectories managed by the InputTrajectories class. A trajectory
    is a specific path or sequence of nodes and edges that represents a distinct flow or process
    within the overall graph structure.

    By modeling a trajectory as a directed graph, this class ensures that the directional
    relationships between nodes are maintained, which is essential for accurately representing
    processes that have a clear start and end point or that flow in a particular direction.

    Attributes:
        (Inherits all attributes from networkx.DiGraph)
    """

    def __repr__(self):
        return (
            f"InputTrajectory (trajectory={self.graph[key_trajectories]!r}, "
            f"nodes={self.number_of_nodes()}, edges={self.number_of_edges()})"
        )

    def to_symetrical_multidigraph(self) -> nx.MultiDiGraph:
        # TODO: several places in the code, this should be used instead of manually doing it.
        """Create a nx.MultiDiGraph version by just making each edge two sided.

        Copies the edge attribute for each direction.

        Returns:
            nx.MultiDiGraph: The same graph but the edges are two sided.
        """
        # Create a new MultiDiGraph and copy graph attributes
        symetrical_graph = nx.MultiDiGraph()
        symetrical_graph.graph.update(self.graph)  # Copy graph-level attributes

        # Add all nodes from G to M, preserving attributes
        for node, data in self.nodes(data=True):
            symetrical_graph.add_node(node, **data)

        # Add all edges from G to M, preserving attributes
        # Since we need to make edges bidirectional, we add each edge in both directions
        for u, v, data in self.edges(data=True):
            symetrical_graph.add_edge(u, v, **data)
            symetrical_graph.add_edge(v, u, **data)

        return symetrical_graph

    def get_unique_root(self):
        """Returns the unique root of the directed graph G.

        A root is defined as a node with no incoming edges.

        Raises:
            ValueError: if there are zero or multiple roots.
        """
        # Find all nodes with in-degree 0
        roots = [node for node in self.nodes if self.in_degree(node) == 0]

        if len(roots) == 1:
            return roots[0]
        elif len(roots) == 0:
            raise ValueError("No root found: the graph has no node with in-degree 0.")
        else:
            raise ValueError(f"Multiple roots found: {roots}")

    def decompose_trajectory_method_1(self: nx.DiGraph, min_size: int = 3) -> List[nx.DiGraph]:
        subgraphs = []

        # Identify all branch points (nodes with out_degree >= 2)
        branches = [n for n in self.nodes() if self.out_degree(n) >= 2]

        # Process each branch point to extract subtrees and linear paths
        for bp in branches:
            # Add subtree rooted at the branch point if it meets size criteria
            descendants = nx.descendants(self, bp)
            subtree_nodes = descendants.union({bp})
            if len(subtree_nodes) >= min_size:
                subtree = self.subgraph(subtree_nodes).copy()
                roots = [n for n in subtree.nodes() if subtree.in_degree(n) == 0]
                if len(roots) == 1:
                    subgraphs.append(subtree)

            # Extract linear paths from each successor to the next critical node
            for succ in self.successors(bp):
                path = [bp]
                current = succ
                while True:
                    path.append(current)
                    if self.out_degree(current) != 1:  # Reached a branch point or leaf
                        break
                    next_nodes = list(self.successors(current))
                    if not next_nodes:  # Leaf node
                        break
                    current = next_nodes[0]
                if len(path) >= min_size:
                    sg = self.subgraph(path).copy()
                    sg_roots = [n for n in sg.nodes() if sg.in_degree(n) == 0]
                    if len(sg_roots) == 1:
                        subgraphs.append(sg)

        # Add the longest path in the DAG
        longest_path = nx.dag_longest_path(self)
        if len(longest_path) >= min_size:
            sg = self.subgraph(longest_path).copy()
            sg_roots = [n for n in sg.nodes() if sg.in_degree(n) == 0]
            if len(sg_roots) == 1:
                subgraphs.append(sg)

        # Handle root nodes that are not branch points
        roots = [n for n in self.nodes() if self.in_degree(n) == 0]
        for root in roots:
            if root not in branches:
                descendants = nx.descendants(self, root)
                subtree_nodes = descendants.union({root})
                if len(subtree_nodes) >= min_size:
                    subtree = self.subgraph(subtree_nodes).copy()
                    sg_roots = [n for n in subtree.nodes() if subtree.in_degree(n) == 0]
                    if len(sg_roots) == 1:
                        subgraphs.append(subtree)

        for s in range(len(subgraphs)):
            subgraphs[s].graph[key_trajectories] = subgraphs[s].graph[key_trajectories] + f"_subgraph_{s}"

        return subgraphs

    def decompose_trajectory_method_2(self, min_size: int = 3) -> List[nx.DiGraph]:

        def _process_path(G: nx.DiGraph, path: list, processed: set, min_size: int) -> None:
            """Handle path decomposition with overlapping chunks"""
            # Add full path if valid
            if len(path) >= min_size:
                _add_valid_subgraph(G, path, processed, min_size)

            # Add overlapping windowed chunks
            for i in range(len(path) - min_size + 1):
                chunk = path[i : i + min_size]
                _add_valid_subgraph(G, chunk, processed, min_size)

        def _trace_linear_path(G: nx.DiGraph, start_node: str) -> list:
            """Follow linear chain until next branch point"""
            path = []
            current = start_node
            while True:
                path.append(current)
                successors = list(G.successors(current))
                if len(successors) != 1:
                    break
                current = successors[0]
            return path

        def _add_valid_subgraph(G: nx.DiGraph, nodes: list, processed: set, min_size: int) -> None:
            """Validate and add unique subgraph configurations"""
            if len(nodes) < min_size:
                return

            node_set = frozenset(nodes)
            if node_set in processed:
                return

            # Validate subgraph properties
            sg = G.subgraph(nodes)
            roots = [n for n in sg.nodes if sg.in_degree(n) == 0]
            if len(roots) == 1 and nx.is_weakly_connected(sg):
                processed.add(node_set)

        # Validation and initialization
        if not nx.is_directed_acyclic_graph(self):
            raise ValueError("Input graph must be a DAG")

        roots = [n for n in self.nodes if self.in_degree(n) == 0]
        if len(roots) != 1:
            raise ValueError("Graph must have exactly one root")
        root = roots[0]

        processed: Set[FrozenSet] = set()

        # 1. Hierarchical functional modules
        branch_points = [n for n in self.nodes if self.out_degree(n) >= 2] + [root]
        for node in branch_points:
            descendants = nx.descendants(self, node).union({node})
            _add_valid_subgraph(self, descendants, processed, min_size)

        # 2. Longest path decomposition
        longest_path = nx.dag_longest_path(self)
        _process_path(self, longest_path, processed, min_size)

        # 3. Branch-to-leaf trajectories
        leaves = [n for n in self.nodes if self.out_degree(n) == 0]
        for leaf in leaves:
            path = nx.shortest_path(self, root, leaf)
            _process_path(self, path, processed, min_size)

        # 4. Stage-wise decomposition
        for bp in branch_points:
            for succ in self.successors(bp):
                path = _trace_linear_path(self, succ)
                if path:
                    full_path = [bp] + path
                    _process_path(self, full_path, processed, min_size)

        # Convert frozen node sets to subgraphs

        subgraphs = [self.subgraph(nodes).copy() for nodes in processed]
        for s in range(len(subgraphs)):
            subgraphs[s].graph[key_trajectories] = subgraphs[s].graph[key_trajectories] + f"_subgraph_{s}"

        return subgraphs

    def plot_trajectory(
        self,
        title="trajectory_name",
        figsize=None,
        font_size=10,
        node_size=300,
        node_color="#8DA0CB",  # Modern muted blue
        cmap="viridis",
        color_nodes_by_position=False,
        # Edge styling parameters
        edge_width=1,
        edge_color="gray",
        color_edges_by_weight=False,
        edge_weight_attribute="weight",
        edge_cmap="plasma",
        edge_vmin=None,
        edge_vmax=None,
        edge_colorbar=True,
        edge_labels=False,
        edge_label_format=".2f",
        edge_label_font_size=8,
        edge_label_color="black",
        edge_label_offset=None,
        # Layout parameters
        label_offset=None,
        title_y=1.02,
        layout_args="-Grankdir=LR -Gnodesep=0.25 -Granksep=0.25",
    ):
        """
        Visualizes the graph with configurable styling and automatic layout adjustments.

        Parameters
        ----------
        title : str
            Title of the plot.
        figsize : tuple, optional
            Figure dimensions (width, height) in inches. If None, size is auto-calculated.
        font_size : int, optional
            Base font size for text elements.
        node_size : int, optional
            Size of the nodes.
        node_color : str or array-like, optional
            Node color or colormap values.
        cmap : str, optional
            Colormap for node coloring when using positional coloring.
        color_nodes_by_position : bool, optional
            Whether to color nodes based on their horizontal position.
        edge_width : int, optional
            Base width for edges.
        edge_color : str or array-like, optional
            Edge color or colormap values.
        color_edges_by_weight : bool, optional
            Whether to color edges by their weight attribute.
        edge_weight_attribute : str, optional
            Edge attribute key for weight values.
        edge_cmap : str, optional
            Colormap for edge coloring when using weight-based coloring.
        edge_vmin : float, optional
            Minimum value for edge colormap scaling.
        edge_vmax : float, optional
            Maximum value for edge colormap scaling.
        edge_colorbar : bool, optional
            Whether to show a colorbar for edge weights.
        edge_labels : bool, optional
            Whether to display edge weight labels.
        edge_label_format : str, optional
            Format string for edge weight labels.
        edge_label_font_size : int, optional
            Font size for edge labels.
        edge_label_color : str, optional
            Color for edge labels.
        edge_label_offset : int, optional
            Offset distance for edge labels from edges. If None, auto-calculated.
        label_offset : int, optional
            Vertical offset for node labels. If None, auto-calculated.
        title_y : float, optional
            Vertical position of the title.
        layout_args : str, optional
            Arguments for graphviz layout engine.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)  # Temporary figure to handle layout calculations
        ax = fig.add_subplot(111)

        # Calculate hierarchical layout using graphviz
        pos = nx.nx_agraph.graphviz_layout(self, prog="dot", args=layout_args)

        # Extract coordinate ranges for automatic parameter calculations
        x_coords = [pos[node][0] for node in self.nodes()] if self.nodes() else []
        y_coords = [pos[node][1] for node in self.nodes()] if self.nodes() else []

        # Automatically determine figure size if not specified
        if figsize is None:
            if x_coords and y_coords:
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                width_points = max_x - min_x
                height_points = max_y - min_y
                # Add 20% padding to each side
                padding_x = 0.2 * width_points if width_points != 0 else 100
                padding_y = 0.2 * height_points if height_points != 0 else 100
                # Convert points to inches (assuming 100dpi)
                fig_width = max((width_points + 2 * padding_x) / 180, 8)  # Minimum width 8"
                fig_height = max((height_points + 2 * padding_y) / 100, 4)  # Minimum height 4"
                figsize = (fig_width, fig_height)
            else:
                figsize = (4, 3)  # Fallback for empty graphs
        plt.close(fig)  # Close temporary figure

        # Create actual figure with determined size
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Node coloring logic
        if color_nodes_by_position:
            try:
                min_x, max_x = min(x_coords), max(x_coords)
                color_vals = [(x - min_x) / (max_x - min_x) if max_x != min_x else 0.5 for x in x_coords]
                node_color = plt.get_cmap(cmap)(color_vals)
            except Exception as e:
                print(f"Warning: Positional node coloring failed - {e}. Using base color.")

        # Edge coloring logic
        if color_edges_by_weight:
            weights = [self[u][v].get(edge_weight_attribute, 1.0) for u, v in self.edges]
            edge_vmin = edge_vmin if edge_vmin is not None else (min(weights) if weights else 0)
            edge_vmax = edge_vmax if edge_vmax is not None else (max(weights) if weights else 1)
            norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
            cmap_edge = plt.get_cmap(edge_cmap)
            edge_colors = [cmap_edge(norm(w)) for w in weights]
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_edge)
            sm.set_array(weights)
        else:
            edge_colors = edge_color
            sm = None

        # Label positioning offsets
        if label_offset is None:
            label_offset = 0

        if edge_label_offset is None:
            edge_label_offset = 0

        # Calculate final label positions
        label_pos = {n: (x, y + label_offset) for n, (x, y) in pos.items()}

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            self, pos, ax=ax, node_size=node_size, node_color=node_color, edgecolors="black", linewidths=0.8, alpha=1.0
        )

        nx.draw_networkx_edges(
            self,
            pos,
            ax=ax,
            arrowstyle="-|>",
            arrowsize=15,
            edge_color=edge_colors,
            width=edge_width,
            alpha=1.0,
            connectionstyle="arc3",
        )

        # Edge label placement
        if edge_labels:
            for u, v in self.edges():
                if edge_weight_attribute not in self[u][v]:
                    continue
                weight = self[u][v][edge_weight_attribute]
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                dx, dy = x2 - x1, y2 - y1
                length = (dx**2 + dy**2) ** 0.5
                if length == 0:
                    continue  # Skip self-loops
                # Calculate perpendicular offset position
                mid_x = (x1 + x2) / 2 + (dy / length) * edge_label_offset
                mid_y = (y1 + y2) / 2 - (dx / length) * edge_label_offset
                ax.text(
                    mid_x,
                    mid_y,
                    f"{weight:{edge_label_format}}",
                    fontsize=edge_label_font_size,
                    color=edge_label_color,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
                )

        # Node labels
        nx.draw_networkx_labels(
            self,
            label_pos,
            ax=ax,
            font_size=font_size,
            font_weight="normal",
            font_family=plt.rcParams["font.family"],
            alpha=1.0,
        )

        # Edge weight colorbar
        if color_edges_by_weight and edge_colorbar and sm:
            cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.3, aspect=40, pad=0.05)
            cbar.set_label("Edge Weight", fontsize=font_size)
            cbar.ax.tick_params(labelsize=font_size - 2)

        # Title styling
        if title == "trajectory_name":
            title = self.graph[key_trajectories]

        fig.suptitle(title, y=title_y, fontsize=font_size + 2, fontweight="bold", color="#333333")

        # Final layout adjustments
        plt.subplots_adjust(
            left=0.03, right=0.97, top=0.97, bottom=0.15 if (color_edges_by_weight and edge_colorbar) else 0.03
        )
        ax.set_axis_off()
        plt.show()

    @cached_property
    def identify(self) -> List[Dict[str, Any]]:
        """Identifies and categorizes structural elements within the trajectory graph.

        This advanced method performs a meticulous and comprehensive analysis of the trajectory's
        structure to identify a wide range of graph elements, including but not limited to:
        linear paths, loops, bifurcations, convergences, trees, and complex networks.
        The method operates recursively to detect both fundamental and composite structures, assigning
        hierarchical levels to each identified subgraph based on their complexity and interconnections.

        The identification process encompasses the following steps:

        1. **Basic Structure Detection**:
            - **Linear Path**: A sequence of nodes where each intermediate node has exactly one
              predecessor and one successor.
            - **Loop/Cycle**: A closed path where the start and end nodes are identical.
            - **Bifurcation**: A node with multiple outgoing edges, indicating a split in the path.
            - **Convergence**: A node with multiple incoming edges, indicating a merge in the path.
            - **Self-loop**: An edge that connects a node to itself.
            - **Isolated Node**: A node with no incoming or outgoing edges.

        2. **Composite Structure Detection**:
            - **Tree**: A hierarchical structure with a single root node and multiple branching paths.
            - **Directed Acyclic Graph (DAG)**: A graph with no cycles, allowing for complex branching.
            - **Complex Network**: An intricate interconnection of multiple substructures, including
              loops, bifurcations, convergences, and trees.

        3. **Hierarchical Level Assignment**:
            - **Level 1**: Fundamental structures (e.g., linear paths, loops, bifurcations, convergences).
            - **Level 2**: Intermediate structures composed of Level 1 elements (e.g., trees, DAGs).
            - **Level 3+**: More complex structures combining multiple Level 2 elements (e.g., complex networks).

        4. **Recursive Analysis**:
            - The method recursively decomposes the graph into subgraphs, ensuring that nested or
              overlapping structures are thoroughly identified and categorized.

        5. **Edge Case Handling**:
            - The method meticulously handles edge cases, such as single-node graphs, multiple
              self-loops, parallel structures, and disconnected components.

        Returns:
            List[Dict[str, Any]]: A comprehensive list of dictionaries, each representing an identified
            subgraph with the following keys:
                - `type` (str): The type of the subgraph (e.g., 'linear_path', 'loop', 'bifurcation').
                - `nodes` (List[str]): The list of nodes constituting the subgraph.
                - `edges` (List[Tuple[str, str]]): The list of edges constituting the subgraph.
                - `level` (int): The hierarchical level of the subgraph.
                - `details` (Dict[str, Any]): Additional details pertinent to the subgraph type, such as
                  cycle length, number of branches, etc.
        """
        identified_elements: List[Dict[str, Any]] = []
        visited_nodes: Set[str] = set()

        def find_linear_path(start_node: str, subgraph: nx.DiGraph) -> List[str]:
            """Finds a maximal linear path starting from the given node.

            Args:
                start_node (str): The node from which to start the linear path.
                subgraph (nx.DiGraph): The subgraph being analyzed.

            Returns:
                List[str]: A list of nodes representing the linear path.
            """
            path = [start_node]
            current_node = start_node

            while True:
                successors = list(subgraph.successors(current_node))
                if len(successors) != 1:
                    break
                next_node = successors[0]
                if subgraph.in_degree(next_node) != 1 or next_node in path:
                    break
                path.append(next_node)
                current_node = next_node

            return path

        def detect_self_loops(subgraph: nx.DiGraph) -> List[str]:
            """Detects all self-loops in the subgraph.

            Args:
                subgraph (nx.DiGraph): The subgraph being analyzed.

            Returns:
                List[str]: A list of nodes that have self-loops.
            """
            return list(nx.nodes_with_selfloops(subgraph))

        def detect_loops(subgraph: nx.DiGraph) -> List[List[str]]:
            """Detects all simple cycles (loops) within the subgraph.

            Args:
                subgraph (nx.DiGraph): The subgraph being analyzed.

            Returns:
                List[List[str]]: A list of loops, each represented as a list of nodes.
            """
            return list(nx.simple_cycles(subgraph))

        def detect_bifurcations(subgraph: nx.DiGraph) -> List[str]:
            """Identifies all bifurcation points within the subgraph.

            Args:
                subgraph (nx.DiGraph): The subgraph being analyzed.

            Returns:
                List[str]: A list of nodes that are bifurcation points.
            """
            return [node for node in subgraph.nodes() if subgraph.out_degree(node) > 1 and node not in visited_nodes]

        def detect_convergences(subgraph: nx.DiGraph) -> List[str]:
            """Identifies all convergence points within the subgraph.

            Args:
                subgraph (nx.DiGraph): The subgraph being analyzed.

            Returns:
                List[str]: A list of nodes that are convergence points.
            """
            return [node for node in subgraph.nodes() if subgraph.in_degree(node) > 1 and node not in visited_nodes]

        def assign_level(element_type: str) -> int:
            """Assigns a hierarchical level based on the element type.

            Args:
                element_type (str): The type of the subgraph element.

            Returns:
                int: The hierarchical level corresponding to the element type.
            """
            hierarchy = {
                "self_loop": 1,
                "linear_path": 1,
                "loop": 1,
                "bifurcation": 2,
                "convergence": 2,
                "tree": 3,
                "dag": 3,
                "complex_network": 4,
            }
            return hierarchy.get(element_type, 0)

        def analyze_subgraph(subgraph: nx.DiGraph, current_level: int):
            """Analyzes a subgraph to identify its structural elements.

            Args:
                subgraph (nx.DiGraph): The subgraph being analyzed.
                current_level (int): The current hierarchical level.
            """
            nonlocal identified_elements, visited_nodes

            # 1. Detect and record self-loops
            self_loops = detect_self_loops(subgraph)
            for node in self_loops:
                if node not in visited_nodes:
                    identified_elements.append(
                        {
                            "type": "self_loop",
                            "nodes": [node],
                            "edges": [(node, node)],
                            "level": assign_level("self_loop"),
                            "details": {},
                        }
                    )
                    visited_nodes.add(node)

            # 2. Detect and record loops/cycles
            loops = detect_loops(subgraph)
            for loop in loops:
                loop_set = set(loop)
                if not loop_set.issubset(visited_nodes):
                    loop_edges = list(zip(loop, loop[1:] + [loop[0]]))
                    identified_elements.append(
                        {
                            "type": "loop",
                            "nodes": loop,
                            "edges": loop_edges,
                            "level": assign_level("loop"),
                            "details": {"cycle_length": len(loop)},
                        }
                    )
                    visited_nodes.update(loop_set)
                    # Recursively analyze the loop
                    loop_subgraph = subgraph.subgraph(loop).copy()
                    analyze_subgraph(loop_subgraph, current_level + 1)

            # 3. Detect and record bifurcations
            bifurcations = detect_bifurcations(subgraph)
            for bifurcation in bifurcations:
                if bifurcation not in visited_nodes:
                    successors = list(subgraph.successors(bifurcation))
                    bifurcation_edges = [(bifurcation, succ) for succ in successors]
                    identified_elements.append(
                        {
                            "type": "bifurcation",
                            "nodes": [bifurcation] + successors,
                            "edges": bifurcation_edges,
                            "level": assign_level("bifurcation"),
                            "details": {"number_of_branches": len(successors)},
                        }
                    )
                    visited_nodes.add(bifurcation)
                    # Recursively analyze each branch
                    for succ in successors:
                        branch_path = find_linear_path(succ, subgraph)
                        if branch_path:
                            branch_subgraph = subgraph.subgraph(branch_path).copy()
                            analyze_subgraph(branch_subgraph, current_level + 1)

            # 4. Detect and record convergences
            convergences = detect_convergences(subgraph)
            for convergence in convergences:
                if convergence not in visited_nodes:
                    predecessors = list(subgraph.predecessors(convergence))
                    convergence_edges = [(pred, convergence) for pred in predecessors]
                    identified_elements.append(
                        {
                            "type": "convergence",
                            "nodes": predecessors + [convergence],
                            "edges": convergence_edges,
                            "level": assign_level("convergence"),
                            "details": {"number_of_incoming": len(predecessors)},
                        }
                    )
                    visited_nodes.add(convergence)
                    # Recursively analyze each incoming path
                    for pred in predecessors:
                        incoming_path = find_linear_path(pred, subgraph)
                        if incoming_path:
                            incoming_subgraph = subgraph.subgraph(incoming_path).copy()
                            analyze_subgraph(incoming_subgraph, current_level + 1)

            # 5. Detect and record linear paths
            for node in subgraph.nodes():
                if node not in visited_nodes:
                    if subgraph.in_degree(node) <= 1 and subgraph.out_degree(node) <= 1:
                        linear_path = find_linear_path(node, subgraph)
                        if len(linear_path) > 1:
                            linear_edges = list(zip(linear_path, linear_path[1:]))
                            identified_elements.append(
                                {
                                    "type": "linear_path",
                                    "nodes": linear_path,
                                    "edges": linear_edges,
                                    "level": assign_level("linear_path"),
                                    "details": {"length": len(linear_path) - 1},
                                }
                            )
                            visited_nodes.update(linear_path)

            # 6. Detect and record trees and DAGs
            # A tree is a connected acyclic graph with one root. A DAG is a directed graph with no cycles.
            if nx.is_directed_acyclic_graph(subgraph):
                # Check if it's a tree (exactly one node with in-degree 0)
                roots = [n for n, d in subgraph.in_degree() if d == 0]
                if len(roots) == 1:
                    identified_elements.append(
                        {
                            "type": "tree",
                            "nodes": list(subgraph.nodes()),
                            "edges": list(subgraph.edges()),
                            "level": assign_level("tree"),
                            "details": {"root": roots[0], "size": len(subgraph.nodes())},
                        }
                    )
                    visited_nodes.update(subgraph.nodes())
                else:
                    # It's a DAG but not a tree
                    identified_elements.append(
                        {
                            "type": "dag",
                            "nodes": list(subgraph.nodes()),
                            "edges": list(subgraph.edges()),
                            "level": assign_level("dag"),
                            "details": {"number_of_roots": len(roots), "size": len(subgraph.nodes())},
                        }
                    )
                    visited_nodes.update(subgraph.nodes())

            # 7. Detect and record complex networks
            # If the subgraph is neither a tree nor a DAG, classify it as a complex network
            if not nx.is_directed_acyclic_graph(subgraph) and not nx.is_tree(subgraph):
                identified_elements.append(
                    {
                        "type": "complex_network",
                        "nodes": list(subgraph.nodes()),
                        "edges": list(subgraph.edges()),
                        "level": assign_level("complex_network"),
                        "details": {"size": len(subgraph.nodes()), "edge_count": len(subgraph.edges())},
                    }
                )
                visited_nodes.update(subgraph.nodes())

        def find_disconnected_components(graph: nx.DiGraph) -> List[nx.DiGraph]:
            """Finds all weakly connected components in the graph as separate subgraphs.

            Args:
                graph (nx.DiGraph): The graph to analyze.

            Returns:
                List[nx.DiGraph]: A list of subgraphs, each representing a weakly connected component.
            """
            return [graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)]

        # 0. Handle completely disconnected nodes (isolated nodes)
        isolated_nodes = list(nx.isolates(self))
        for node in isolated_nodes:
            identified_elements.append(
                {
                    "type": "isolated_node",
                    "nodes": [node],
                    "edges": [],
                    "level": (
                        assign_level("isolated_node")
                        if "isolated_node"
                        in {"self_loop", "linear_path", "loop", "bifurcation", "convergence", "isolated_node"}
                        else 0
                    ),
                    "details": {},
                }
            )
            visited_nodes.add(node)

        # 1. Decompose the graph into weakly connected components to handle disconnected graphs
        components = find_disconnected_components(self)
        for component in components:
            # Skip if the component is a single isolated node (already handled)
            if len(component.nodes()) == 1 and list(component.edges()) == []:
                continue
            analyze_subgraph(component, current_level=1)

        return identified_elements

    def define_basic_elements(self) -> Dict[str, List[Dict[str, Any]]]:
        """Defines the graph using only basic structural elements.

        This method leverages the comprehensive `identify` property to detect various structural
        elements within the trajectory graph. It filters out non-basic elements, such as
        composite structures like trees and complex networks, and focuses solely on fundamental
        components. The basic elements include:
            - `self_loop`
            - `linear_path`
            - `loop`
            - `bifurcation`
            - `convergence`
            - `isolated_node`

        By isolating these basic elements, this method provides a simplified and foundational
        representation of the graph, facilitating tasks that require understanding or manipulating
        the graph at its most elementary level.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary where each key corresponds to a basic
                element type (e.g., 'linear_path', 'loop', 'bifurcation'), and each value is a list
                of dictionaries representing the identified elements of that type. Each element
                dictionary contains:
                    - `nodes` (List[str]): The list of nodes constituting the subgraph.
                    - `edges` (List[Tuple[str, str]]): The list of edges constituting the subgraph.
                    - `level` (int): The hierarchical level of the subgraph.
                    - `details` (Dict[str, Any]): Additional details pertinent to the subgraph type.

        Raises:
            ValueError: If no basic structural elements are identified in the graph.
        """
        # Define the set of basic element types
        basic_types = {"self_loop", "linear_path", "loop", "bifurcation", "convergence", "isolated_node"}

        # Invoke the identify property to get all identified elements
        identified = self.identify

        # Initialize a dictionary to hold basic elements
        basic_elements: Dict[str, List[Dict[str, Any]]] = {element_type: [] for element_type in basic_types}

        # Iterate through all identified elements and filter out non-basic types
        for element in identified:
            element_type = element["type"]
            if element_type in basic_types:
                # Remove the 'type' key as it's now the dictionary key
                element_copy = element.copy()
                del element_copy["type"]
                basic_elements[element_type].append(element_copy)

        # Remove keys with empty lists to clean up the dictionary
        basic_elements = {k: v for k, v in basic_elements.items() if v}

        if not basic_elements:
            raise ValueError("No basic structural elements identified in the graph.")

        return basic_elements

    # Separate getter methods for each structure type

    def get_self_loops(self) -> List[Dict[str, Any]]:
        """Retrieves all self-loop structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing self-loop subgraphs.
        """
        return [element for element in self.identify if element["type"] == "self_loop"]

    def get_linear_paths(self) -> List[Dict[str, Any]]:
        """Retrieves all linear path structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing linear path subgraphs.
        """
        return [element for element in self.identify if element["type"] == "linear_path"]

    def get_loops(self) -> List[Dict[str, Any]]:
        """Retrieves all loop (cycle) structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing loop subgraphs.
        """
        return [element for element in self.identify if element["type"] == "loop"]

    def get_bifurcations(self) -> List[Dict[str, Any]]:
        """Retrieves all bifurcation structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing bifurcation subgraphs.
        """
        return [element for element in self.identify if element["type"] == "bifurcation"]

    def get_convergences(self) -> List[Dict[str, Any]]:
        """Retrieves all convergence structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing convergence subgraphs.
        """
        return [element for element in self.identify if element["type"] == "convergence"]

    def get_isolated_nodes(self) -> List[Dict[str, Any]]:
        """Retrieves all isolated node structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing isolated node subgraphs.
        """
        return [element for element in self.identify if element["type"] == "isolated_node"]

    def get_trees(self) -> List[Dict[str, Any]]:
        """Retrieves all tree structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing tree subgraphs.
        """
        return [element for element in self.identify if element["type"] == "tree"]

    def get_dags(self) -> List[Dict[str, Any]]:
        """Retrieves all Directed Acyclic Graph (DAG) structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing DAG subgraphs.
        """
        return [element for element in self.identify if element["type"] == "dag"]

    def get_complex_networks(self) -> List[Dict[str, Any]]:
        """Retrieves all complex network structures within the trajectory graph.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing complex network subgraphs.
        """
        return [element for element in self.identify if element["type"] == "complex_network"]
