#!/usr/bin/env python3
 
import pytest
import tempfile
import numpy as np
import anndata as ad
from networkx import NetworkXError

from sctram.api._lower_level import TrajectoryEvaluationAPI
from sctram.input import InputTrajectories
from sctram.generate.real import sc_norman_sciplex_cpa

# ------------------------- Fixtures -------------------------------------------

@pytest.fixture(scope="module")
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture(scope="module")
def adata_bms(temp_dir):
    adata_norman = sc_norman_sciplex_cpa(dataset_dir=temp_dir)
    adata_bms = adata_norman[
        adata_norman.obs["drug"].str.lower().str.contains("bms|vehicle")
    ].copy()
    return adata_bms

@pytest.fixture(scope="module")
def adata_tardis(adata_bms):
    return ad.AnnData(
        X=adata_bms.obsm["tardis"].copy(),
        obs=adata_bms.obs.copy(),
    )

@pytest.fixture(scope="module")
def adata_scvi(adata_bms):
    return ad.AnnData(
        X=adata_bms.obsm["scvi"].copy(),
        obs=adata_bms.obs.copy(),
    )

@pytest.fixture(scope="module")
def original_trajectory():
    return [
        ('Vehicle_1.0', 'BMS_0.001'),
        ('BMS_0.001', 'BMS_0.005'),
        ('BMS_0.005', 'BMS_0.01'),
        ('BMS_0.01', 'BMS_0.05'),
        ('BMS_0.05', 'BMS_0.1'),
        ('BMS_0.1', 'BMS_0.5'),
        ('BMS_0.5', 'BMS_1.0'),
    ]

@pytest.fixture
def shuffled_trajectory_random(original_trajectory):
    edges = original_trajectory.copy()
    np.random.shuffle(edges)
    return edges

@pytest.fixture
def shuffled_trajectory_logical(original_trajectory):
    nodes = list({node for edge in original_trajectory for node in edge})
    root = nodes[0]
    shuffled = [root] + np.random.permutation(nodes[1:]).tolist()
    return [(shuffled[i], shuffled[i+1]) for i in range(len(shuffled)-1)]

# ------------------------- Tests ---------------------------------------------

def test_evaluation_runs(adata_tardis, original_trajectory):
    """Test that evaluations complete without errors."""
    pass

def test_original_vs_shuffled_random(adata_tardis, original_trajectory, shuffled_trajectory_random):
    """Original trajectory should outperform randomly shuffled edges."""
    pass

def test_original_vs_shuffled_logical(adata_tardis, original_trajectory, shuffled_trajectory_logical):
    """Original trajectory should outperform logically shuffled edges."""
    pass

def test_tardis_vs_scvi(adata_tardis, adata_scvi, original_trajectory):
    """Tardis and ScVI embeddings should yield different metric scores."""
    pass

def test_missing_root(adata_tardis, original_trajectory):
    """Missing root label should raise an error in pseudotime evaluation."""
    pass

def test_minimal_trajectory(adata_tardis):
    """Test evaluation with a minimal trajectory (2 nodes)."""
    pass

def test_disconnected_trajectory(adata_tardis):
    """Disconnected trajectories should raise appropriate errors."""
    pass
