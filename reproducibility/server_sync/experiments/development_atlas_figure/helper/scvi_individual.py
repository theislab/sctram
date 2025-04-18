print("started")
import os
import sys
import gc
import argparse
import warnings
import anndata as ad
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np
import scvi
sc.settings.verbose = 3
print("libraries loaded")

def main():
    print("arg parsing")
    parser = argparse.ArgumentParser(description="blabla.")
    parser.add_argument("--handle_anndata", type=str, required=True)
    parser.add_argument("--adata_file_path", type=str, required=True)
    parser.add_argument("--trained_model_path", type=str, required=True)
    parser.add_argument("--trained_latent_path", type=str, required=True)
    args = parser.parse_args()

    print("anndata prep")
    adata_file_path = args.adata_file_path
    trained_model_path = args.trained_model_path
    trained_latent_path = args.trained_latent_path
    handle_anndata = args.handle_anndata
    
    adata = ad.read_h5ad(adata_file_path)
    adata = adata[adata.obs["handle_anndata"] == handle_anndata]
    adata = adata.copy()
    gc.collect()
    print(adata)

    limit_batches = min(1.0, 200000/len(adata))
    print(f"limit_batches: {limit_batches}")

    model_params = dict(
        n_latent=24, 
        gene_likelihood = "nb",
    )
    
    train_params = dict(
        train_size=0.8,
        batch_size=512,
        limit_train_batches=limit_batches, 
        limit_val_batches=limit_batches,
        max_epochs=400,
        plan_kwargs=dict(
            n_steps_kl_warmup = None,
            n_epochs_kl_warmup = 400
        ),
    )
    
    dataset_params = dict(
        layer=None, 
        labels_key=None,
        batch_key="concatenated_integration_covariates",
        categorical_covariate_keys=None,
    )
    
    scvi.model.SCVI.setup_anndata(adata, **dataset_params)
    
    vae = scvi.model.SCVI(adata, **model_params)
    vae.train(**train_params)

    print("training finished.")
    vae.save(trained_model_path, overwrite=True)

    print("latent calculate")
    latent = ad.AnnData(X=vae.get_latent_representation(), obs=adata.obs.copy())
    sc.pp.neighbors(latent, n_neighbors=50)
    sc.tl.umap(latent)
    print("latent write")
    latent.write_h5ad(trained_latent_path)
    
    print("\n\n\nDone!")

if __name__ == "__main__":
    main()
