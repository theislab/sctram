#!/usr/bin/env python3

import numpy as np
import networkx as nx
from collections import Counter

from loguru import logger
_logger = logger.bind(name="BaseMetric")
try:
    from sctram.evaluate._metrics.validators import validate_inclusive_between_0_1 as _validator
except ImportError:
    _logger.warning(f"Validation function not found. Skipping validation: {__file__}")
    def _validator(*args, **kwargs):
        pass
    

def _compute_synchronized_wl_labels(graphs, iterations):
    """Compute synchronized Weisfeiler-Lehman labels for multiple graphs.

    This function is intended for a mathematical/statistical audience and generates
    iterative node labels based on local graph structure. It uses the Weisfeiler-Lehman
    procedure to refine initial labels (node degrees) over a specified number of iterations,
    capturing increasingly complex connectivity patterns.

    Parameters:
        graphs (list of nx.Graph): List of input graphs.
        iterations (int): Number of WL refinement iterations.

    Returns:
        list: A list where each element corresponds to a graph and contains a list of label
              lists for each iteration.
    """
    all_labels = []
    for g in graphs:
        nodes = list(g.nodes())
        # Initialize labels with the degree of each node
        labels = [str(g.degree(node)) for node in nodes]
        all_labels.append([labels.copy()])
    
    for _ in range(iterations):
        all_composites = []
        per_graph_composites = []
        
        # Generate composite labels for all nodes in all graphs
        for g_idx, g in enumerate(graphs):
            current_labels = all_labels[g_idx][-1]
            nodes = list(g.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            composites = []
            for node in nodes:
                neighbors = list(g.neighbors(node))
                neighbor_labels = [current_labels[node_to_idx[n]] for n in neighbors]
                composite = (current_labels[node_to_idx[node]],) + tuple(sorted(neighbor_labels))
                composites.append(composite)
            per_graph_composites.append(composites)
            all_composites.extend(composites)
        
        # Create global unique label mapping
        unique_composites = sorted(set(all_composites), key=lambda x: x)
        comp_to_label = {comp: str(i) for i, comp in enumerate(unique_composites)}
        
        # Update labels for each graph
        for g_idx in range(len(graphs)):
            new_labels = [comp_to_label[c] for c in per_graph_composites[g_idx]]
            all_labels[g_idx].append(new_labels.copy())
    
    return all_labels

def _wl_subtree_kernel(g1, g2, iterations):
    """Compute the Weisfeiler-Lehman subtree kernel between two graphs using synchronized labels.

    This function quantifies similarity between two graphs by comparing the frequency of
    subtree patterns (captured by WL labels) over several iterations.

    Parameters:
        g1 (nx.Graph): First graph.
        g2 (nx.Graph): Second graph.
        iterations (int): Number of WL iterations.

    Returns:
        int: The computed kernel value representing the similarity based on common subtree counts.
    """
    labels = _compute_synchronized_wl_labels([g1, g2], iterations)
    labels1, labels2 = labels[0], labels[1]
    
    kernel = 0
    for i in range(iterations + 1):
        count1 = Counter(labels1[i])
        count2 = Counter(labels2[i])
        kernel += sum(count1[k] * count2.get(k, 0) for k in count1)
    return kernel

def weisfeiler_lehman_distance(given_adjacency_matrix: np.ndarray, 
                                  inferred_adjacency_matrix: np.ndarray, 
                                  validate_result: bool,
                                  threshold: float, 
                                  iterations: int = 3, 
                                  remove_self_loops: bool = True
                                  ) -> float:
    """Compute the normalized Weisfeiler-Lehman subtree kernel distance between two square adjacency matrices.

    This method compares two square adjacency matrices by first binarizing them using a given threshold,
    optionally removing self-loops, and then converting them into NetworkX graphs. The similarity between 
    the graphs is measured using the Weisfeiler-Lehman (WL) subtree kernel, which counts common subtree patterns 
    across iterative label refinements. The Weisfeiler-Lehman subtree kernel is computed over a specified number 
    of iterations. This involves iterative label refinements that capture hierarchical substructure patterns
    within the graphs. A normalized similarity measure is derived by comparing the cross-kernel (between the two 
    graphs) with the self-kernels (each graph compared with itself). The final distance is calculated as one minus this 
    normalized similarity, yielding a value in the range [0, 1].
    
    Parameters:
        given_adjacency_matrix (np.ndarray): The reference square adjacency matrix.
        inferred_adjacency_matrix (np.ndarray): The inferred square adjacency matrix.
        threshold (float, optional): Binarization threshold to convert weighted edges into binary edges. 
        iterations (int, optional): Number of WL iterations for label refinement. Default is 3.
        remove_self_loops (bool, optional): If True, diagonal entries (self-loops) are set to 0. Default is True.
        validate_result (bool, optional): If True, validates that the computed distance is zero or positive. 

    Returns:
        float: The normalized WL subtree kernel distance in the range [0, 1]. 
               A score of 0 indicates identical graphs, while higher scores denote greater dissimilarity.

    Advantages:
        - Provides a graph comparison method that captures global structural patterns.
        - Sensitive to both edge additions and deletions by leveraging WL subtree patterns.
        - Normalization allows comparisons across graphs of the same size.

    Limitations:
        - It is primarily focused on topological isomorphism
        - The method may be sensitive to the choice of threshold and the number of WL iterations.
        - Does not localize differences to specific nodes or substructures.
        - Assumes that both input matrices are square and represent undirected graphs.

    Interpretation:
        - A distance of 0 indicates that the graphs are structurally identical.
        - Lower values imply high similarity, while values approaching 1 indicate significant structural differences.
    """
    # Binarize matrices and remove self-loops
    inferred_binary = (inferred_adjacency_matrix >= threshold).astype(int)
    given_binary = (given_adjacency_matrix >= threshold).astype(int)
    
    if remove_self_loops:
        np.fill_diagonal(given_binary, 0)
        np.fill_diagonal(inferred_binary, 0)
    
    # Convert to NetworkX graphs
    g_given = nx.from_numpy_array(given_binary)
    g_inferred = nx.from_numpy_array(inferred_binary)

    # Compute kernel values with edge case handling
    kernel = _wl_subtree_kernel(g_given, g_inferred, iterations)
    kernel_given = _wl_subtree_kernel(g_given, g_given, iterations)
    kernel_inferred = _wl_subtree_kernel(g_inferred, g_inferred, iterations)
    
    # print(kernel)
    # print(kernel_given)
    # print(kernel_inferred)

    # Calculate normalized similarity with numerical stability
    # Ensures scores are interpretable regardless of trajectory size or density.
    denominator = np.sqrt(kernel_given * kernel_inferred) + 1e-12
    similarity = kernel / denominator
    score = max(0.0, 1.0 - similarity)

    if validate_result:
        _validator(score=score)
    
    return score


def _run_tests():
    # --------------------------------------------------------------------------
    # Test 1: Identical Graphs
    # --------------------------------------------------------------------------
    # Two identical small graphs should yield a distance exactly 0.
    adj_identical = np.array([[0, 1, 0],
                              [1, 0, 0],
                              [0, 0, 0]])
    distance = weisfeiler_lehman_distance(adj_identical.copy(), 
                                             adj_identical.copy(), 
                                             threshold=0.5, iterations=3, 
                                             remove_self_loops=True, validate_result=True)
    # print("Test 1 (Identical graphs):", distance)
    # Expectation: distance should be exactly 0 (or numerically close to 0)
    assert np.isclose(distance, 0.0, atol=1e-8), f"Test 1 failed: Expected 0.0, got {distance}"

    # --------------------------------------------------------------------------
    # Test 2: Completely Different Graphs (Empty vs. Fully Connected)
    # --------------------------------------------------------------------------
    # An empty graph compared with a fully connected graph (after self-loops removed)
    # should yield the maximum distance, i.e., 1.0.
    adj_empty = np.zeros((3, 3))
    adj_complete = np.ones((3, 3)) - np.eye(3)
    distance = weisfeiler_lehman_distance(adj_empty.copy(), 
                                             adj_complete.copy(), 
                                             threshold=0.5, iterations=3, 
                                             remove_self_loops=True, validate_result=True)
    # print("Test 2 (Empty vs. fully connected):", distance)
    # Expectation: distance should be 1.0 (or extremely close due to normalization)
    assert np.isclose(distance, 1.0, atol=1e-8), f"Test 2 failed: Expected 1.0, got {distance}"

    # --------------------------------------------------------------------------
    # Test 3: Cycle Graph vs. Path Graph
    # --------------------------------------------------------------------------
    # A cycle graph (closed loop) and a path graph (open chain) on the same number of nodes
    # should have a measurable difference. Here we expect a non-zero distance that is not extreme.
    cycle = nx.cycle_graph(10)
    path = nx.path_graph(10)
    A_cycle = nx.to_numpy_array(cycle)
    A_path = nx.to_numpy_array(path)
    distance = weisfeiler_lehman_distance(A_cycle.copy(), A_path.copy(), 
                                             threshold=0.5, iterations=5, 
                                             remove_self_loops=True, validate_result=True)
    # print("Test 3 (Cycle vs. Path):", distance)
    # We expect a moderate distance, for example between 0.2 and 0.8.
    assert 0.2 < distance < 0.8, f"Test 3 failed: Expected distance in (0.2, 0.8), got {distance}"

    # --------------------------------------------------------------------------
    # Test 4a: Weighted Graphs Around Threshold
    # --------------------------------------------------------------------------
    # Two weighted graphs where edge weights are around the threshold may lead to subtle differences.
    # By fixing the threshold at 0.5, some edges are binarized differently.
    A_weighted1 = np.array([
        [0.0, 0.49, 0.80],
        [0.49, 0.0, 0.51],
        [0.80, 0.51, 0.0]
    ])
    A_weighted2 = np.array([
        [0.0, 0.51, 0.80],
        [0.51, 0.0, 0.49],
        [0.80, 0.49, 0.0]
    ])

    distance = weisfeiler_lehman_distance(A_weighted1, A_weighted2, 
                                            threshold=0.5, iterations=3, 
                                            remove_self_loops=True, validate_result=True)
    # print("Test 4a isomorphic (Weighted graphs with threshold=0.5):", distance)

    assert np.isclose(distance, 0.0, atol=1e-8), f"Test 4a failed: Expected distance > 0.00, got {distance}"
    
    # After binarization at threshold 0.5, both A_weighted1 and A_weighted2 produce graphs that are structurally 
    # identical (isomorphic) through node relabeling. The WL kernel cannot distinguish between isomorphic graphs, 
    # correctly returning a distance of 0.0, which violates the test's assumption of non-isomorphism.
    
    # --------------------------------------------------------------------------
    # Test 4b: Weighted Graphs Around Threshold
    # --------------------------------------------------------------------------
    # Two weighted graphs where edge weights are around the threshold may lead to subtle differences.
    # By fixing the threshold at 0.5, some edges are binarized differently.
    A_weighted1 = np.array([
        [0.0, 0.49, 0.80],
        [0.49, 0.0, 0.51],
        [0.80, 0.51, 0.0]
    ])
    A_weighted2 = np.array([
        [0.0, 0.51, 0.80],
        [0.51, 0.0, 0.51],  # Edge between 1-2 added (0.51 >= 0.5)
        [0.80, 0.51, 0.0]
    ])

    distance = weisfeiler_lehman_distance(A_weighted1, A_weighted2, 
                                            threshold=0.5, iterations=3, 
                                            remove_self_loops=True, validate_result=True)
    
    # print("Test 4b nonisomorphic (Weighted graphs with threshold=0.5):", distance)
    assert distance > 0.05, f"Test 4b failed: Expected distance > 0.05, got {distance}"

    # --------------------------------------------------------------------------
    # Test 5: Community Structure Graphs (Stochastic Block Model)
    # --------------------------------------------------------------------------
    # Generate two graphs using the stochastic block model with identical parameters.
    # The community structure should lead to a relatively low distance even if the realizations differ.
    sizes = [15, 15]
    probs = [[0.8, 0.2],
             [0.2, 0.8]]
    sbm1 = nx.stochastic_block_model(sizes, probs, seed=44)
    sbm2 = nx.stochastic_block_model(sizes, probs, seed=45)
    A_sbm1 = nx.to_numpy_array(sbm1)
    A_sbm2 = nx.to_numpy_array(sbm2)
    distance = weisfeiler_lehman_distance(A_sbm1.copy(), A_sbm2.copy(), 
                                             threshold=0.5, iterations=4, 
                                             remove_self_loops=True, validate_result=True)
    # print("Test 5 (Community structure graphs):", distance)
    # Expect a moderate distance, but lower than the completely dissimilar graphs.
    # For example, we expect the distance to be below 0.6.
    assert distance < 0.6, f"Test 5 failed: Expected distance < 0.6, got {distance}"

    # --------------------------------------------------------------------------
    # Test 6: Effect of Self-Loop Removal
    # --------------------------------------------------------------------------
    # For a graph with only self-loops vs. one with no edges, when self-loops are removed the distance should be 0.
    # When self-loops are not removed, the self-kernel differences should lead to a non-zero distance.
    A_self_loops = np.eye(2)  # Graph with only self-loops
    A_no_edges = np.zeros((2, 2))
    distance_removed = weisfeiler_lehman_distance(A_self_loops.copy(), A_no_edges.copy(), 
                                                     threshold=0.5, remove_self_loops=True, validate_result=True)
    distance_kept = weisfeiler_lehman_distance(A_self_loops.copy(), A_no_edges.copy(), 
                                                  threshold=0.5, remove_self_loops=False, validate_result=True)
    # print("Test 6 (Self-loop removal): removed =", distance_removed, "kept =", distance_kept)
    # When self-loops are removed, both graphs become empty so the distance should be 0.
    assert np.isclose(distance_removed, 0.0, atol=1e-8), f"Test 6 failed: Expected 0.0 with self-loops removed, got {distance_removed}"
    # When self-loops are kept, the difference in self-loop presence should lead to a distance noticeably above 0.
    assert distance_kept > 0.0, f"Test 6 failed: Expected distance > 0 with self-loops kept, got {distance_kept}"

    # --------------------------------------------------------------------------
    # Test 7: Sensitivity to WL Iterations
    # --------------------------------------------------------------------------
    # For two similar graphs, increasing the number of WL iterations should capture deeper structural differences.
    # We check that the computed distance changes (and typically increases) as we increase the iterations.
    A_cycle = nx.to_numpy_array(nx.cycle_graph(8))
    A_path = nx.to_numpy_array(nx.path_graph(8))
    distances = []
    for iters in range(2, 7):
        d = weisfeiler_lehman_distance(A_cycle.copy(), A_path.copy(), 
                                          threshold=0.5, iterations=iters, 
                                          remove_self_loops=True, validate_result=True)
        distances.append(d)
        # print(f"Test 7 (Iterations = {iters}): distance = {d}")
    # Check that there is a significant change (at least 0.05 difference) between the minimum and maximum computed distances.
    if len(distances) > 1:
        assert max(distances) - min(distances) >= 0.05, "Test 7 failed: Distance not sensitive to iteration changes"

    # --------------------------------------------------------------------------
    # Test 8: Isomorphic Graphs with Permuted Nodes
    # --------------------------------------------------------------------------
    # Create two isomorphic graphs with permuted node order
    adj1 = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0]
    ])
    # Permute nodes [0,1,2,3] -> [0,1,3,2]
    adj2 = adj1[[0, 1, 3, 2], :][:, [0, 1, 3, 2]]
    distance = weisfeiler_lehman_distance(adj1, adj2, threshold=0.5, iterations=3, 
                                         remove_self_loops=True, validate_result=True)
    # print("Test 8 (Permuted isomorphic graphs):", distance)
    assert np.isclose(distance, 0.0, atol=1e-8), f"Test 8 failed: Expected 0.0, got {distance}"

    # --------------------------------------------------------------------------
    # Test 9: Non-isomorphic Graphs Indistinguishable by WL
    # --------------------------------------------------------------------------
    # Two triangles vs. a hexagon (WL cannot distinguish)
    adj_two_triangles = np.zeros((6, 6))
    adj_two_triangles[0:3, 0:3] = nx.to_numpy_array(nx.cycle_graph(3))
    adj_two_triangles[3:6, 3:6] = nx.to_numpy_array(nx.cycle_graph(3))
    adj_hexagon = nx.to_numpy_array(nx.cycle_graph(6))
    distance = weisfeiler_lehman_distance(adj_two_triangles, adj_hexagon, threshold=0.5,
                                         iterations=3, remove_self_loops=True, validate_result=True)
    # print("Test 9 (WL-indistinguishable graphs):", distance)
    assert np.isclose(distance, 0.0, atol=1e-8), f"Test 9 failed: Expected 0.0, got {distance}"

    # --------------------------------------------------------------------------
    # Test 10: Different-sized Adjacency Matrices
    # --------------------------------------------------------------------------
    adj_2x2 = np.zeros((2, 2))
    adj_3x3 = np.zeros((3, 3))
    try:
        distance = weisfeiler_lehman_distance(adj_2x2, adj_3x3, threshold=0.5, iterations=3,
                                              remove_self_loops=True, validate_result=True)
        # print("Test 10 (Different-sized matrices): No error. Distance:", distance)
    except Exception as e:
        # print("Test 10 (Different-sized matrices): Error raised -", e)
        assert False, "Function should handle different-sized matrices without error"

    # --------------------------------------------------------------------------
    # Test 11: Directed Edges Converted to Undirected
    # --------------------------------------------------------------------------
    adj_directed1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    adj_directed2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    distance = weisfeiler_lehman_distance(adj_directed1, adj_directed2, threshold=0.5,
                                         iterations=3, remove_self_loops=True, validate_result=True)
    
    distance = weisfeiler_lehman_distance(adj_directed1, adj_directed2, threshold=0.5,
                                        iterations=3, remove_self_loops=True, validate_result=True)
    # print("Test 11 (Directed to undirected isomorphic):", distance, " expected:", 0.0, "[skipped]")
    # Weisfeiler-Lehman (WL) kernel is not designed to handle directed graphs. 
    # assert np.isclose(distance, 0.0, atol=1e-8), f"Test 11 failed: Expected 0.0, got {distance}"

    # --------------------------------------------------------------------------
    # Test 12: Single Edge vs. Empty Graph
    # --------------------------------------------------------------------------
    adj_empty = np.zeros((2, 2))
    adj_one_edge = np.array([[0, 1], [1, 0]])
    distance = weisfeiler_lehman_distance(adj_empty, adj_one_edge, threshold=0.5, iterations=1,
                                         remove_self_loops=True, validate_result=True)
    # print("Test 12 (Empty vs. one edge):", distance)
    assert np.isclose(distance, 1.0, atol=1e-8), f"Test 12 failed: Expected 1.0, got {distance}"

    # --------------------------------------------------------------------------
    # Test 13: Normalization Check with Manual Calculation
    # --------------------------------------------------------------------------
    g1 = nx.Graph()
    g1.add_edge(0, 1)
    adj1 = nx.to_numpy_array(g1)
    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2)])
    adj2 = nx.to_numpy_array(g2)
    distance = weisfeiler_lehman_distance(adj1, adj2, threshold=0.5, iterations=1,
                                         remove_self_loops=True, validate_result=True)
    expected_similarity = 4 / (np.sqrt(8 * 10))
    expected_distance = 1 - expected_similarity
    # print("Test 13 (Normalization check):", distance, "Expected:", expected_distance)
    assert np.isclose(distance, expected_distance, atol=1e-4), f"Test 13 failed: Expected {expected_distance}, got {distance}"

    print("All tests passed.")

if __name__ == "__main__":
    _run_tests()
