#!/usr/bin/env python3

import logging
import os

import networkx as nx
import pytest
import yaml

# Import your evaluation class (adjust the import path as needed)
from sctram.evaluate._adjacency import AdjacencyMatrixEvaluation

# =============================================================================
# Helper functions
# =============================================================================


def load_test_cases():
    """Load the YAML file with test cases."""
    test_file = os.path.join(os.path.dirname(__file__), "resources", "trajectories_test.yaml")
    with open(test_file) as f:
        test_cases = yaml.safe_load(f)
    return test_cases["test_cases"]


def create_graph_from_edges(edges):
    """
    Given a list of edge pairs, create and return a NetworkX graph.
    (Use DiGraph or Graph depending on your package requirements.)
    """
    G = nx.DiGraph()
    G.add_edges_from([tuple(edge) for edge in edges])
    return G


def create_dummy_labels(graph):
    """
    Create a dummy labels dictionary mapping each node to itself.
    In your real application labels might have more meaning.
    """
    return {node: node for node in graph.nodes()}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def test_cases():
    return load_test_cases()


@pytest.fixture
def evaluation_instance():
    # List all metrics that you want to test.
    metrics = [
        "frobenius",
        "L1",
        "accuracy",
        "graph_edit_distance",
        "spectral_distance",
        "jaccard",
        "hamming",
        "precision",
        "recall",
        "f1_score",
        "mantel_correlation",
        "ssim",
        "avg_shortest_path_diff",
        "degree_emd",
        "clustering_coeff_diff",
        "graphlet_degree_vector",
        "weisfeiler_lehman_distance",
        "gnn_embedding_distance",
        "persistence_diagram_distance",
        "maximum_common_subgraph_distance",
        "random_walk_kernel_distance",
    ]
    return AdjacencyMatrixEvaluation(method_params={"metrics": metrics})


# =============================================================================
# Test Functions
# =============================================================================


@pytest.mark.parametrize("case", load_test_cases())
def test_trajectory_metrics(case, evaluation_instance):
    """
    For each test case defined in the YAML file, compare the scores computed
    for a perfect (ground truth) trajectory and an imperfect one.
    """
    # Create the graphs from YAML definitions.
    gt_graph = create_graph_from_edges(case["ground_truth"]["edges"])
    perfect_graph = create_graph_from_edges(case["trajectories"]["perfect"]["edges"])

    # Here we assume a test case defines an alternative trajectory (e.g., 'imperfect' or 'linear').
    # You can extend this loop to compare several variants if needed.
    for test_variant, trajectory_def in case["trajectories"].items():
        if test_variant == "perfect":
            continue  # we use perfect as the baseline

        variant_graph = create_graph_from_edges(trajectory_def["edges"])
        labels = create_dummy_labels(gt_graph)

        # Evaluate metrics for the perfect trajectory vs. ground truth.
        evaluation_instance.evaluate(given_trajectory=gt_graph, inferred_trajectory=perfect_graph, labels=labels)
        results_perfect = evaluation_instance.get_result()

        # Evaluate metrics for the test (imperfect) trajectory vs. ground truth.
        evaluation_instance.evaluate(given_trajectory=gt_graph, inferred_trajectory=variant_graph, labels=labels)
        results_variant = evaluation_instance.get_result()

        # Depending on whether lower scores indicate a better match (or vice-versa),
        # compare the results. (You might want to add per–metric rules if needed.)
        lower_better = case.get("expected_ordering", {}).get("lower_better", True)
        for metric, perfect_score in results_perfect.items():
            variant_score = results_variant.get(metric)
            # Log the scores for debugging purposes.
            logging.info(
                f"Test case '{case['name']}', variant '{test_variant}', metric '{metric}': perfect={perfect_score} vs variant={variant_score}"
            )
            if lower_better:
                assert perfect_score <= variant_score, (
                    f"For metric '{metric}', the perfect trajectory should have a lower (better) score. "
                    f"Found: perfect={perfect_score}, variant={variant_score}"
                )
            else:
                assert perfect_score >= variant_score, (
                    f"For metric '{metric}', the perfect trajectory should have a higher (better) score. "
                    f"Found: perfect={perfect_score}, variant={variant_score}"
                )
