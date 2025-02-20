
working_directory = "/Users/kemalinecik/git_nosync/sctram"
import sys
sys.path.append(working_directory)

import logging
import os
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import anndata as ad

from sctram.generate.real import sc_norman_sciplex_cpa
from sctram.api._lower_level import TrajectoryEvaluationAPI
from sctram.input import read_dict

dataset_dir = "/Users/kemalinecik/git_nosync/sctram/__temp__/data"
adata_norman = sc_norman_sciplex_cpa(dataset_dir=dataset_dir)
adata_bms = adata_norman[["bms" in i.lower() or "vehicle" in i.lower() for i in adata_norman.obs["drug"]]]

adata_tardis = ad.AnnData(X=adata_bms.obsm["tardis"].copy(), obs=adata_bms.obs.copy())
adata_scvi = ad.AnnData(X=adata_bms.obsm["scvi"].copy(), obs=adata_bms.obs.copy())

ground_truth_trajectories = {
    "trajectory_1": [
        ('Vehicle_1.0', 'BMS_0.001'),
        ('BMS_0.001', 'BMS_0.005'),
        ('BMS_0.005', 'BMS_0.01'),
        ('BMS_0.01', 'BMS_0.05'),
        ('BMS_0.05', 'BMS_0.1'),
        ('BMS_0.1', 'BMS_0.5'),
        ('BMS_0.5', 'BMS_1.0'),
    ],
}
input_trajectories_all = read_dict(ground_truth_trajectories)
input_trajectories = input_trajectories_all.get_trajectory("trajectory_1", include_additional_nodes=False)

api_tardis = TrajectoryEvaluationAPI(
    adata=adata_tardis,
    input_trajectories=input_trajectories,
    labels_obs="drug_dose_name",
    root_label="Vehicle_1.0",
    logger_level="DEBUG"
)
api_scvi = TrajectoryEvaluationAPI(
    adata=adata_scvi,
    input_trajectories=input_trajectories,
    labels_obs="drug_dose_name",
    root_label="Vehicle_1.0",
    logger_level="WARNING"
)

api_tardis.evaluate_with_defaults()
api_scvi.evaluate_with_defaults()