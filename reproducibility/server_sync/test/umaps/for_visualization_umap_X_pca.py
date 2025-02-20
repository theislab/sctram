# working_directory = "/Users/kemalinecik/git_nosync/sctram"
working_directory = "/home/icb/kemal.inecik/work/codes/sctram"

import sys
sys.path.append(working_directory)

import logging
import gc
import os
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import anndata as ad

sc.settings.verbose = 3

from sctram.generate.real import sc_suo_developmental_complete

obsm_key = 'X_pca'
dataset_dir = "/home/icb/kemal.inecik/lustre_workspace/temp_sctram_data"

adata_file_name_with_umap = os.path.join(dataset_dir, f"suo_with_umap_{obsm_key}.h5ad")
adata_ = sc_suo_developmental_complete(dataset_dir=dataset_dir)
print("adata_")
print(adata_)
adata_ = adata_[adata_.obs["LVL0"] == "Haematopoeitic_lineage"].copy()
adata_ = adata_[adata_.obs["LVL1"].isin(["MEM", "Haematopoetic_progenitors"])].copy()
adata = ad.AnnData(X=adata_.obsm[obsm_key], obs=adata_.obs.copy())
print("adata")
print(adata)
gc.collect()

sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.umap(adata)

adata.write_h5ad(adata_file_name_with_umap)
