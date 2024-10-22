#!/usr/bin/env python3

from functools import cached_property
from typing import Any, Dict, List, Set

import networkx as nx


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

    def to_symetrical_multidigraph(self) -> nx.MultiDiGraph:
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
