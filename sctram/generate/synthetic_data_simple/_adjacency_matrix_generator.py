#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class AdjacencyMatrixGenerator:

    @staticmethod
    def generate_complex_adjacency_matrix_with_labels(
        total_nodes: int,
        structures: List[Dict],
        connected_weight_range: Tuple[float, float] = (0.7, 1.0),
        associated_weight_range: Tuple[float, float] = (0.4, 0.6),
        unconnected_weight_range: Tuple[float, float] = (0.0, 0.1),
        inter_structure_connection_prob: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Generate a complex weighted adjacency matrix with specified graph structures, labels, and association-based edge weights.

        Args:
            total_nodes (int): Total number of nodes in the graph.
            structures (list of dict): List of structures to include. Each dict should have:
                - 'name' (str): Unique identifier for the structure.
                - 'type' (str): Type of the structure ('loop', 'linear', 'bifurcation', 'star', 'tree', 'grid', etc.).
                - 'num_nodes' (int): Number of nodes in this structure.
                - 'associated' (list of str): List of names of structures this structure is associated with.
                - Additional parameters depending on the structure type.
            connected_weight_range (tuple, optional): Range for weights of intra-structure edges.
            associated_weight_range (tuple, optional): Range for weights of inter-structure edges between associated structures.
            unconnected_weight_range (tuple, optional): Range for weights of edges not within or between associated structures.
            inter_structure_connection_prob (float, optional): Probability of connecting nodes between associated structures.
            seed (int, optional): Seed for random number generators for reproducibility. Defaults to None.

        Returns:
            adjacency_matrix (np.ndarray): The generated weighted adjacency matrix with weights between 0 and 1.
            labels (list of str): Labels indicating the structure each node belongs to (e.g., 'loop', 'linear', etc.).
        """
        if seed is not None:
            np.random.seed(seed)

        # Validate input
        total_structure_nodes = sum(structure["num_nodes"] for structure in structures)
        if total_structure_nodes > total_nodes:
            raise ValueError(
                f"Total nodes required by structures ({total_structure_nodes}) exceed total_nodes ({total_nodes})."
            )

        # Initialize the graph
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))  # Nodes are labeled from 0 to total_nodes - 1

        labels = ["other"] * total_nodes  # Initialize all labels as 'other'

        current_node = 0  # Pointer to assign nodes

        # Mapping from structure name to its node indices
        structure_name_to_nodes = {}

        for structure in structures:
            struct_name = structure.get("name")
            struct_type = structure.get("type", "").lower()
            num_nodes = structure.get("num_nodes")

            if not struct_name:
                raise ValueError("Each structure must have a 'name' key.")

            if struct_type not in {"loop", "linear", "bifurcation", "star", "tree", "grid"}:
                raise ValueError(f"Unsupported structure type: {struct_type}")

            if struct_type == "loop":
                if num_nodes < 3:
                    raise ValueError("A loop must have at least 3 nodes.")
                subgraph_nodes = list(range(current_node, current_node + num_nodes))
                subg = nx.cycle_graph(n=num_nodes)
                mapping = {i: node for i, node in enumerate(subgraph_nodes)}
                subg = nx.relabel_nodes(subg, mapping)
                G.add_edges_from(subg.edges())
                for node in subgraph_nodes:
                    labels[node] = "loop"
                structure_name_to_nodes[struct_name] = subgraph_nodes
                current_node += num_nodes

            elif struct_type == "linear":
                if num_nodes < 2:
                    raise ValueError("A linear structure must have at least 2 nodes.")
                subgraph_nodes = list(range(current_node, current_node + num_nodes))
                subg = nx.path_graph(n=num_nodes)
                mapping = {i: node for i, node in enumerate(subgraph_nodes)}
                subg = nx.relabel_nodes(subg, mapping)
                G.add_edges_from(subg.edges())
                for node in subgraph_nodes:
                    labels[node] = "linear"
                structure_name_to_nodes[struct_name] = subgraph_nodes
                current_node += num_nodes

            elif struct_type == "bifurcation":
                # Bifurcation: one root node branching into multiple paths
                branches = structure.get("branches", 2)  # Default to 2 branches
                min_nodes = 1 + branches  # At least 1 root + 1 per branch
                if num_nodes < min_nodes:
                    raise ValueError(f"A bifurcation must have at least {min_nodes} nodes for {branches} branches.")
                subgraph_nodes = list(range(current_node, current_node + num_nodes))
                root = subgraph_nodes[0]
                G.add_node(root)
                labels[root] = "bifurcation_root"
                nodes_per_branch = (num_nodes - 1) // branches
                for b in range(branches):
                    start = 1 + b * nodes_per_branch
                    end = start + nodes_per_branch
                    if b == branches - 1:
                        # Last branch takes the remaining nodes
                        end = num_nodes
                    branch_nodes = subgraph_nodes[start:end]
                    if len(branch_nodes) == 0:
                        continue
                    G.add_edge(root, branch_nodes[0])
                    for i in range(len(branch_nodes) - 1):
                        G.add_edge(branch_nodes[i], branch_nodes[i + 1])
                    for node in branch_nodes:
                        labels[node] = "bifurcation_branch"
                structure_name_to_nodes[struct_name] = subgraph_nodes
                current_node += num_nodes

            elif struct_type == "star":
                if num_nodes < 2:
                    raise ValueError("A star structure must have at least 2 nodes.")
                subgraph_nodes = list(range(current_node, current_node + num_nodes))
                center = subgraph_nodes[0]
                leaves = subgraph_nodes[1:]
                for leaf in leaves:
                    G.add_edge(center, leaf)
                labels[center] = "star_center"
                for leaf in leaves:
                    labels[leaf] = "star_leaf"
                structure_name_to_nodes[struct_name] = subgraph_nodes
                current_node += num_nodes

            elif struct_type == "tree":
                # Simple balanced binary tree
                depth = structure.get("depth", 3)
                expected_nodes = sum([2**i for i in range(depth + 1)])
                if num_nodes < expected_nodes:
                    raise ValueError(
                        f"A balanced binary tree of depth {depth} requires at least {expected_nodes} nodes."
                    )
                subgraph_nodes = list(range(current_node, current_node + num_nodes))
                subg = nx.balanced_tree(r=2, h=depth)
                if subg.number_of_nodes() > num_nodes:
                    raise ValueError(
                        f"Binary tree with depth {depth} exceeds the number of nodes specified ({num_nodes})."
                    )
                mapping = {i: node for i, node in enumerate(subgraph_nodes)}
                subg = nx.relabel_nodes(subg, mapping)
                G.add_edges_from(subg.edges())
                for node in subgraph_nodes:
                    labels[node] = "tree"
                structure_name_to_nodes[struct_name] = subgraph_nodes
                current_node += num_nodes

            elif struct_type == "grid":
                # Grid structure (2D grid)
                rows = structure.get("rows", 3)
                cols = structure.get("cols", 3)
                required_nodes = rows * cols
                if num_nodes != required_nodes:
                    raise ValueError(f"For grid structure, num_nodes must equal rows * cols ({rows * cols}).")
                subgraph_nodes = list(range(current_node, current_node + num_nodes))
                subg = nx.grid_2d_graph(rows, cols)
                mapping = {(i, j): subgraph_nodes[i * cols + j] for i, j in subg.nodes()}
                subg = nx.relabel_nodes(subg, mapping)
                G.add_edges_from(subg.edges())
                for node in subgraph_nodes:
                    labels[node] = "grid"
                structure_name_to_nodes[struct_name] = subgraph_nodes
                current_node += num_nodes

            else:
                raise ValueError(f"Unsupported structure type: {struct_type}")

        # Initialize the adjacency matrix with unconnected weights
        adjacency_matrix = np.random.uniform(
            low=unconnected_weight_range[0], high=unconnected_weight_range[1], size=(total_nodes, total_nodes)
        )
        # Make the matrix symmetric by copying the upper triangle to the lower triangle
        adjacency_matrix = np.triu(adjacency_matrix)  # Upper triangle
        adjacency_matrix += adjacency_matrix.T - np.diag(adjacency_matrix.diagonal())  # Mirror upper to lower

        # Assign weights to intra-structure edges
        for u, v in G.edges():
            weight = np.random.uniform(low=connected_weight_range[0], high=connected_weight_range[1])
            adjacency_matrix[u, v] = weight
            adjacency_matrix[v, u] = weight  # Ensure symmetry

        # Assign weights to inter-structure edges based on associations
        processed_pairs = set()  # To avoid processing the same pair twice
        for structure in structures:
            struct_name = structure["name"]
            associated_structures = structure.get("associated", [])
            struct_nodes = structure_name_to_nodes.get(struct_name, [])

            for assoc_struct_name in associated_structures:
                # Avoid processing the same pair twice
                pair = tuple(sorted([struct_name, assoc_struct_name]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                assoc_struct_nodes = structure_name_to_nodes.get(assoc_struct_name, [])
                if not assoc_struct_nodes:
                    continue  # No nodes in the associated structure

                # Connect nodes between struct_nodes and assoc_struct_nodes based on inter_structure_connection_prob
                for u in struct_nodes:
                    for v in assoc_struct_nodes:
                        if u == v:
                            continue  # Skip self-loop
                        if np.random.rand() < inter_structure_connection_prob:
                            weight = np.random.uniform(low=associated_weight_range[0], high=associated_weight_range[1])
                            adjacency_matrix[u, v] = weight
                            adjacency_matrix[v, u] = weight  # Ensure symmetry

        np.fill_diagonal(adjacency_matrix, 0.0)  # No self-loops

        return adjacency_matrix, labels

    @staticmethod
    def generate_random_adjacency_matrix(graph_type="erdos_renyi", num_nodes=100, weighted=True, **kwargs):
        """
        Generate a complex adjacency matrix for testing purposes with weights between 0 and 1.

        Args:
            graph_type (str): The type of graph to generate. Options are:
                - 'erdos_renyi': Random graph
                - 'barabasi_albert': Scale-free network
                - 'watts_strogatz': Small-world network
                - 'stochastic_block': Graph with community structure
                - 'random_regular': Regular graph
                - 'powerlaw_cluster': Power law cluster graph
                - 'custom': Provide your own NetworkX graph via kwargs['graph']
            num_nodes (int): Number of nodes in the graph
            weighted (bool): If True, assign random weights between 0 and 1 to edges.
            **kwargs: Additional keyword arguments specific to the graph type.

        Returns:
            np.ndarray: Adjacency matrix of the generated graph with weights between 0 and 1
        """
        if graph_type == "erdos_renyi":
            p = kwargs.get("p", 0.05)  # Probability for edge creation
            g = nx.erdos_renyi_graph(n=num_nodes, p=p, seed=kwargs.get("seed", None))
        elif graph_type == "barabasi_albert":
            m = kwargs.get("m", max(1, num_nodes // 100))  # Number of edges to attach from a new node
            g = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=kwargs.get("seed", None))
        elif graph_type == "watts_strogatz":
            k = kwargs.get(
                "k", max(2, num_nodes // 10)
            )  # Each node is connected to k nearest neighbors in ring topology
            p = kwargs.get("p", 0.1)  # Probability of rewiring each edge
            g = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p, seed=kwargs.get("seed", None))
        elif graph_type == "stochastic_block":
            sizes = kwargs.get("sizes", [num_nodes // 3] * 3)
            probs = kwargs.get("probs", [[0.1, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.1]])
            g = nx.stochastic_block_model(sizes, probs, seed=kwargs.get("seed", None))
        elif graph_type == "random_regular":
            d = kwargs.get("d", max(2, num_nodes // 10))  # Degree of each node
            if d * num_nodes % 2 != 0:
                raise ValueError("For random_regular_graph, d*num_nodes must be even.")
            g = nx.random_regular_graph(d=d, n=num_nodes, seed=kwargs.get("seed", None))
        elif graph_type == "powerlaw_cluster":
            m = kwargs.get("m", max(1, num_nodes // 100))  # Number of edges to attach from a new node
            p = kwargs.get("p", 0.1)  # Probability of adding a triangle after adding a random edge
            g = nx.powerlaw_cluster_graph(n=num_nodes, m=m, p=p, seed=kwargs.get("seed", None))
        elif graph_type == "custom":
            g = kwargs.get("graph", None)
            if g is None:
                raise ValueError("For 'custom' graph type, provide a NetworkX graph via 'graph' keyword argument.")
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        if weighted:
            # Assign random weights between 0 and 1 to each edge
            for u, v in g.edges():
                g[u][v]["weight"] = np.random.uniform(0.0, 1.0)
        else:
            # For unweighted graphs, ensure all existing edges have weight 1.0
            for u, v in g.edges():
                g[u][v]["weight"] = 1.0

        # Convert to adjacency matrix with weights
        if weighted:
            # Use the 'weight' attribute for edge weights
            adjacency_matrix = nx.to_numpy_array(g, weight="weight")
        else:
            # If unweighted, use the 'weight' attribute set to 1.0
            adjacency_matrix = nx.to_numpy_array(g, weight=None)
            adjacency_matrix[adjacency_matrix != 0] = 1.0  # Ensure binary weights

        # Ensure the adjacency matrix is of type float and values are between 0 and 1
        adjacency_matrix = np.array(adjacency_matrix, dtype=float)
        adjacency_matrix = np.clip(adjacency_matrix, 0.0, 1.0)

        return adjacency_matrix
