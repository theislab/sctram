print("started")

print("loading libraries.")
import argparse
import os
print("loading libraries..")
import anndata as ad
import torch
import scvi
print("libraries loaded.")

print(f"CUDA used: {torch.cuda.is_available()}")
dataset_dir = "/home/icb/kemal.inecik/lustre_workspace/temp_sctram_data"

def main():
    parser = argparse.ArgumentParser(description="blabla.")
    parser.add_argument("--epoch", type=int, required=True, help="Number of epoch to run the script")
    parser.add_argument("--model_str", type=str, required=True, help="which model to run.")
    parser.add_argument("--overwrite", type=lambda x: x.lower() in ['true', '1', 'yes'], required=True, help="Set to True to overwrite existing files, False otherwise.")
    args = parser.parse_args()

    print(f"Training for epoch {args.epoch!r} for Suo dataset {args.model_str} training.")
    
    output_dir_path = os.path.join(dataset_dir, f"model_suo_incremental_training_{args.model_str}_epoch_{int(args.epoch)}")
    overwrite = bool(args.overwrite)
    if overwrite:
        print("override set to `True`")
    else:
        assert not os.path.isdir(output_dir_path) and not os.path.exists(output_dir_path)
        print("override set to `False`")
    
    print("dataset loading.")
    adata_file_path = os.path.join("/home/icb/kemal.inecik/lustre_workspace/tardis_data/processed", "dataset_complete_Suo.h5ad")
    assert os.path.isfile(adata_file_path), f"File not already exist: `{adata_file_path}`"
    adata = ad.read_h5ad(adata_file_path)
    adata.obs["age"] = adata.obs["age"].astype("str").astype("category")
    print(adata)
    print("dataset loaded.")

    expected_size = 841922
    assert len(adata) == expected_size, "step calculation should be dependent on adata size"
    # step_per_epoch = 841922 * 0.25 / 512 # note that 512 batch_size, 0.25 limit_train_batches
    
    model_params = dict(
        n_latent=24, 
        gene_likelihood = "nb",
    )

    train_params = dict(
        train_size=0.8,
        batch_size=512,
        limit_train_batches=0.25, 
        limit_val_batches=0.25,
        plan_kwargs=dict(
            n_epochs_kl_warmup=400    
        ),
        max_steps=int(args.epoch),
        max_epochs=None
    )
    
    if args.model_str == "scvi":
    
        dataset_params = dict(
            layer=None, 
            batch_key="concatenated_integration_covariates",
            categorical_covariate_keys=None,
        )

        scvi.model.SCVI.setup_anndata(adata, **dataset_params)
        print("training starting.")
        vae = scvi.model.SCVI(adata, **model_params)
        vae.train(**train_params)
        print("training finished.")

    elif args.model_str == "scanvi":
    
        dataset_params = dict(
            layer=None, 
            labels_key="cell_type",
            unlabeled_category="NA",
            batch_key="concatenated_integration_covariates",
            categorical_covariate_keys=None,
        )
    
        scvi.model.SCANVI.setup_anndata(adata, **dataset_params)
        print("training starting.")
        vae = scvi.model.SCANVI(adata, **model_params)
        vae.train(**train_params)
        print("training finished.")

    else:
        raise ValueError(f"model_str: {args.model_str!r}")

    vae.save(output_dir_path, overwrite=overwrite)
    print(f"Saved result for epoch {args.epoch!r} for Suo dataset {args.model_str!r} training to {output_dir_path!r}")

if __name__ == "__main__":
    main()

