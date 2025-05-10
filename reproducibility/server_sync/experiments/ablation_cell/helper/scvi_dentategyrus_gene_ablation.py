import sys

working_directory = "/home/icb/kemal.inecik/work/codes/sctram"
sys.path.append(working_directory)
sys.path.append("/home/icb/kemal.inecik/work/codes/sctram/reproducibility/server_sync/experiments")

import argparse
import copy
import gc
import itertools
import os
import subprocess
import sys
import pickle
import scvi
import time
import warnings
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
import torch
from sklearn.neighbors import kneighbors_graph

from ablation_cell.helper.random_remove import remove_random_fraction_of_genes

print(f"CUDA used: {torch.cuda.is_available()}")

dataset_dir = "/home/icb/kemal.inecik/lustre_workspace/temp_sctram_data"
helpers_directory = os.path.join(os.getcwd(), "helper")
logs_directory = os.path.join(os.getcwd(), "logs")

def main():
    parser = argparse.ArgumentParser(description="blabla.")
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    
    args = parser.parse_args()
    fraction = args.fraction
    seed = args.seed
    
    definitions_path = os.path.join(dataset_dir, f"incremental_analysis_definitions_dict.pkl")
    with open(definitions_path, "rb") as _file_definitions:
        definitions = pickle.load(_file_definitions)  

    adata = ad.read_h5ad(definitions["scvi_dentategyrus"]["training_anndata_path"])
    adata.X = adata.layers["counts"].copy()
    del adata.layers["counts"]
    gc.collect()

    adata = remove_random_fraction_of_genes(adata, k=fraction, seed=seed)

    model_params = dict(
        n_latent=24, 
        gene_likelihood = "nb",
    )
    
    train_params = dict(
        train_size=0.8,
        batch_size=512,
        limit_train_batches=1.00, 
        limit_val_batches=1.00,
        max_epochs=400,
    )
    
    dataset_params = dict(
        layer=None, 
        # labels_key="cell_type",
        # unlabeled_category="NA",
        batch_key="SampleID",
        categorical_covariate_keys=None,
    )
    
    scvi.model.SCVI.setup_anndata(adata, **dataset_params)
    vae = scvi.model.SCVI(adata, **model_params)
    vae.train(**train_params)  # callbacks=[logger_cb]

    model_dir_path = os.path.join(dataset_dir, f"scvi_dentategyrus_ablation_gene_fraction_{fraction}_seed{seed}")
    vae.save(
        model_dir_path,
        overwrite=True,
    )

    latent = ad.AnnData(X=vae.get_latent_representation())
    latent.obs = adata.obs.copy()
    sc.pp.neighbors(latent)
    sc.tl.umap(latent)

    latent_dir_path = os.path.join(dataset_dir, f"scvi_dentategyrus_ablation_gene_fraction_{fraction}_seed{seed}.h5ad")
    latent.write_h5ad(latent_dir_path)

    
    
    
if __name__ == "__main__":
    main()
        
    
    
    
    
    
    
    
    
    
