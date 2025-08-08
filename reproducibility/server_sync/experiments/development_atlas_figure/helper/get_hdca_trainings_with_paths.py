# From `hdca/integration/4.0_training/benchmarking/collect.ipynb` at 09.04.2025

import os
import pandas as pd
import pickle
import sqlite3
import numpy as np

def get_hdca_trainings_with_paths():
    trained_runs_path_benchmarked = "/lustre/groups/ml01/workspace/kemal.inecik/hdca/temp/models/trained_models_benchmarked.pickle"
    with open(trained_runs_path_benchmarked, 'rb') as file:
        arguments, runs_df = pickle.load(file)
    
    scvi_runs = "/home/icb/kemal.inecik/work/codes/hdca/integration/4.0_training/scvi/trained_hypers.db"
    scpoli_runs = "/home/icb/kemal.inecik/work/codes/hdca/integration/4.0_training/scpoli/trained_hypers.db"
    with sqlite3.connect(scvi_runs) as conn:
        trained_scvi = pd.read_sql_query("SELECT * FROM data", conn)
    with sqlite3.connect(scpoli_runs) as conn:
        trained_scpoli = pd.read_sql_query("SELECT * FROM data", conn)
    
    dataset_dict = {
        '/lustre/groups/ml01/workspace/kemal.inecik/hdca/temp/preprocessing/unification_union_20240330_hvg_integration.h5ad': "union",
        '/lustre/groups/ml01/workspace/kemal.inecik/hdca/temp/preprocessing/unification_union_20240330_hvg-intersection_integration.h5ad': "intersection",
    }
    
    _run_df = []
    for ind, run in runs_df.iterrows():
        if run["model"] in ["pca", "harmony"]:
            run["n_epochs_kl_warmup"] = np.nan
            run["batch_strategy"] = "concatenate"
            run["dataset_strategy"] = dataset_dict[item["dataset"].item()]
        elif run["model"] == "scvi":
            item = trained_scvi[trained_scvi["experiment_id"] == run["name"]]
            assert len(item) == 1
            run["n_epochs_kl_warmup"] = item["n_epochs_kl_warmup"].item()
            run["batch_strategy"] = item["batch_strategy"].item()
        elif run["model"] == "scpoli":
            item = trained_scpoli[trained_scpoli["experiment_id"] == run["name"]]
            assert len(item) == 1
            run["n_epochs_kl_warmup"] = item["n_epochs_kl_warmup"].item()
            run["batch_strategy"] = item["batch_strategy"].item()
        elif run["model"] in ["tardis", "scanoroma"]:
            continue
        else:
            raise ValueError
        
        run["dataset_strategy"] = dataset_dict[run["dataset"]]
        for label_key_level in ["LVL2", "LVL3"]:
            run["label_key"] = label_key_level
            _run_df.append(run.copy())
    
    __run_df = []
    for run in _run_df:
        pickle_path_bio = os.path.join("/home/icb/kemal.inecik/lustre_workspace/hdca/temp/scib/20240619", f"{run['name']}_bio_{run['label_key']}.pickle")
        pickle_path_batch = os.path.join("/home/icb/kemal.inecik/lustre_workspace/hdca/temp/scib/20240619", f"{run['name']}_batch_{run['label_key']}.pickle")    
        metrics_bio = pd.read_pickle(pickle_path_bio).loc["model"]
        metrics_batch = pd.read_pickle(pickle_path_batch).loc["model"]
        __run_df.append(pd.concat([run, metrics_bio, metrics_batch]))
    df = pd.DataFrame(__run_df)
    df["Total"] = df["Bio conservation"] * 0.6 + df["Batch correction"] * 0.4
    
    df_lvl2 = df[df["label_key"] == "LVL2"]
    df_lvl3 = df[df["label_key"] == "LVL3"]
    df_lvl2 = df_lvl2.sort_values(by=["Total"], ascending=False).copy()
    df_lvl3 = df_lvl3.sort_values(by=["Total"], ascending=False).copy()
    
    return df, df_lvl2, df_lvl3