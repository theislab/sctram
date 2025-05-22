import sys

working_directory = "/home/icb/kemal.inecik/work/codes/sctram"
sys.path.append(working_directory)
sys.path.append("/home/icb/kemal.inecik/work/codes/tardis")

import argparse
import copy
import gc
import itertools
import os
import subprocess
import sys
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

import tardis
from tardis._disentanglementmanager import DisentanglementManager as DM
from sctram.generate.real import sc_norman_sciplex_cpa

tardis.config = tardis.config_server

print(f"CUDA used: {torch.cuda.is_available()}")

dataset_dir = "/home/icb/kemal.inecik/lustre_workspace/temp_sctram_data"
helpers_directory = os.path.join(os.getcwd(), "helper")
logs_directory = os.path.join(os.getcwd(), "logs")

def main():
    parser = argparse.ArgumentParser(description="blabla.")
    parser.add_argument("--scanning_coefficient", type=float, required=True)
    args = parser.parse_args()
    
    scanning_coefficient = args.scanning_coefficient

    print(f"Training for scanning_coefficient {args.scanning_coefficient!r}.")
    
    adata = sc_norman_sciplex_cpa(dataset_dir=dataset_dir)
    
    adata.X = adata.layers["counts"].copy()
    del adata.layers
    adata.obs.loc[adata.obs["dose"] == "0.0", "dose_val"] = 0.0
    d = {i: ind for ind, i in enumerate(sorted(adata.obs["dose"].astype(float).unique()))}
    adata.obs["dose_training"] = [d[float(i)] for i in adata.obs["dose"]]
    gc.collect()
    
    print(adata)
    
    model_level_metrics = [
        dict(
            metric_identifier="metric_mi|dose_training",
            training_set=["train", "validation"],
            every_n_epoch=5,
            subsample=1.0,
            progress_bar=True,
            metric_kwargs=dict(variation="normalized", discretization_bins=256, latent_subset=None, reduce=np.mean),
        ),
        dict(
            metric_identifier="metric_mi|condition",
            training_set=["train", "validation"],
            every_n_epoch=5,
            subsample=1.0,
            progress_bar=True,
            metric_kwargs=dict(variation="normalized", discretization_bins=256, latent_subset=None, reduce=np.mean),
        ),
    ]
    
    warmup_epoch_range = [6, 48]
    dtc_w1 = 100 * scanning_coefficient 
    dtc_w2 = 10 * scanning_coefficient

    dtc_w1_dose = 100 * scanning_coefficient * scanning_coefficient 
    dtc_w2_dose = 10 * scanning_coefficient * scanning_coefficient
    
    counteractive_minibatch_settings = dict(
        method="categorical_random",
        method_kwargs=dict(
            within_labels=False,
            within_batch=False,
            within_categorical_covs=None,
            seed="forward",
        ),
    )
    
    disentenglement_targets_configurations = [
        dict(
            obs_key="dose_training",
            n_reserved_latent=8,
            counteractive_minibatch_settings=counteractive_minibatch_settings,
            auxillary_losses=[
                dict(
                    apply=True,
                    target_type="pseudo_categorical",
                    non_categorical_coefficient_method="squared_difference",
                    progress_bar=False,
                    weight=dtc_w1_dose * 1,
                    method="mse_z",
                    latent_group="reserved",
                    counteractive_example="negative",
                    transformation="inverse",
                    warmup_epoch_range=warmup_epoch_range,
                    method_kwargs={},
                ),
                dict(
                    apply=True,
                    target_type="categorical",
                    progress_bar=False,
                    weight=dtc_w2_dose * 1,
                    method="mse_z",
                    latent_group="reserved",
                    counteractive_example="positive",
                    transformation="none",
                    warmup_epoch_range=warmup_epoch_range,
                    method_kwargs={},
                ),
                dict(
                    apply=True,
                    target_type="categorical",
                    progress_bar=False,
                    weight=dtc_w2_dose * 1,
                    method="mse_z",
                    latent_group="unreserved_complete",
                    counteractive_example="negative",
                    transformation="none",
                    warmup_epoch_range=warmup_epoch_range,
                    method_kwargs={},
                ),
                dict(
                    apply=True,
                    target_type="categorical",
                    progress_bar=False,
                    weight=dtc_w1_dose * 1,
                    method="mse_z",
                    latent_group="unreserved_complete",
                    counteractive_example="positive",
                    transformation="inverse",
                    warmup_epoch_range=warmup_epoch_range,
                    method_kwargs={},
                ),
            ],
        ),
        dict(
            obs_key="condition",
            n_reserved_latent=8,
            counteractive_minibatch_settings=counteractive_minibatch_settings,
            auxillary_losses=[
                dict(
                    apply=True,
                    target_type="categorical",
                    progress_bar=False,
                    weight=dtc_w1,
                    method="mse_z",
                    latent_group="reserved",
                    counteractive_example="negative",
                    transformation="inverse",
                    warmup_epoch_range=warmup_epoch_range,
                    method_kwargs={},
                ),
                dict(
                    apply=True,
                    target_type="categorical",
                    progress_bar=False,
                    weight=dtc_w2,
                    method="mse_z",
                    latent_group="reserved",
                    counteractive_example="positive",
                    transformation="none",
                    warmup_epoch_range=warmup_epoch_range,
                    method_kwargs={},
                ),
                dict(
                    apply=True,
                    target_type="categorical",
                    progress_bar=False,
                    weight=dtc_w2 * 1,
                    method="mse_z",
                    latent_group="unreserved_complete",
                    counteractive_example="negative",
                    transformation="none",
                    warmup_epoch_range=warmup_epoch_range,
                    method_kwargs={},
                ),
                dict(
                    apply=True,
                    target_type="categorical",
                    progress_bar=False,
                    weight=dtc_w1 * 1,
                    method="mse_z",
                    latent_group="unreserved_complete",
                    counteractive_example="positive",
                    transformation="inverse",
                    warmup_epoch_range=warmup_epoch_range,
                    method_kwargs={},
                ),
            ],
        ),
    ]
    
    n_epochs_kl_warmup = 600
    
    model_params = dict(
        n_hidden=512,
        n_layers=3,
        n_latent=(24 + 8 * len(disentenglement_targets_configurations)),
        gene_likelihood="nb",
        use_batch_norm="none",
        use_layer_norm="both",
        dropout_rate=0.5,
        deeply_inject_disentengled_latents=True,
        include_auxillary_loss=True,
        beta_kl_weight=0.5,
        encode_covariates=False,
    )
    
    train_params = dict(
        max_epochs=600,
        train_size=0.8,
        batch_size=64,
        check_val_every_n_epoch=10,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        learning_rate_monitor=True,
        # early stopping:
        early_stopping=False,
        early_stopping_patience=150,
        early_stopping_monitor="elbo_train",
        plan_kwargs=dict(
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            lr=5e-5,
            weight_decay=1e-2,
            optimizer="AdamW",
            # lr-scheduler:
            reduce_lr_on_plateau=True,
            lr_patience=100,
            lr_scheduler_metric="elbo_train",
        ),
    )
    
    dataset_params = dict(
        layer=None,
        labels_key=None,
        batch_key=None,
        categorical_covariate_keys=None,
        disentenglement_targets_configurations=disentenglement_targets_configurations,
        model_level_metrics=model_level_metrics,
        model_level_metrics_helper_covariates=["condition", "dose_training"],
    )
    
    tardis.MyModel.setup_anndata(adata, **dataset_params)
    dataset_params["adata_path"] = "<sctram>"
    dataset_params["adata"] = "sciplex_norman"
    
    tardis.MyModel.setup_wandb(
        wandb_configurations=tardis.config.wandb,
        hyperparams=dict(
            model_params=model_params,
            train_params=train_params,
            dataset_params=dataset_params,
        ),
    )
    
    vae = tardis.MyModel(adata, **model_params)
    vae.train(**train_params)
    
    model_dir_path = os.path.join(dataset_dir, f"tardis_cpa_sciplex_v2_encode_weight_scan_{scanning_coefficient}")
    vae.save(
        model_dir_path,
        overwrite=True,
    )
    
    print("\n\n\n\n")
    print(vae.get_reconstruction_r2_training(top_n=[]))
    
    adata.obs["validation"] = "train"
    adata.obs["validation"].iloc[vae.validation_indices] = "validation"
    
    sublatent = DM.configurations.get_by_obs_key("dose_training").reserved_latent_indices
    latent_dose = ad.AnnData(X=vae.get_latent_representation()[:, sublatent], obs=adata.obs.copy())
    sc.pp.neighbors(latent_dose)
    sc.tl.umap(latent_dose)
    latent_dose.write_h5ad(os.path.join(dataset_dir, f"tardis_cpa_sciplex_v2_encode_weight_scan_{scanning_coefficient}_dose.h5ad"))
    
    sublatent = DM.configurations.get_by_obs_key("condition").reserved_latent_indices
    latent_condition = ad.AnnData(X=vae.get_latent_representation()[:, sublatent], obs=adata.obs.copy())
    sc.pp.neighbors(latent_condition)
    sc.tl.umap(latent_condition)
    latent_condition.write_h5ad(os.path.join(dataset_dir, f"tardis_cpa_sciplex_v2_encode_weight_scan_{scanning_coefficient}_condition.h5ad"))
    
    sublatent = DM.configurations.unreserved_latent_indices
    latent_unreserved = ad.AnnData(X=vae.get_latent_representation()[:, sublatent], obs=adata.obs.copy())
    sc.pp.neighbors(latent_unreserved)
    sc.tl.umap(latent_unreserved)
    latent_unreserved.write_h5ad(os.path.join(dataset_dir, f"tardis_cpa_sciplex_v2_encode_weight_scan_{scanning_coefficient}_unreserved.h5ad"))
    
    latent_whole = ad.AnnData(X=vae.get_latent_representation(), obs=adata.obs.copy())
    sc.pp.neighbors(latent_whole)
    sc.tl.umap(latent_whole)
    latent_whole.write_h5ad(os.path.join(dataset_dir, f"tardis_cpa_sciplex_v2_encode_weight_scan_{scanning_coefficient}_whole.h5ad"))
    
    
if __name__ == "__main__":
    main()
        
    
    
    
    
    
    
    
    
    
