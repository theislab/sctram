print("started")
import argparse
import gc
import os
import pickle
import sys

import anndata as ad
import scanpy as sc

# Add working directory to path
working_directory = "/home/icb/kemal.inecik/work/codes/sctram"
sys.path.append(working_directory)
from sctram.api._lower_level import TrajectoryEvaluationAPI
from sctram.generate.real import sc_suo_developmental_complete
from sctram.input import InputTrajectories

print("libraries loaded")

sc.settings.verbose = 3
dataset_dir = "/home/icb/kemal.inecik/lustre_workspace/temp_sctram_data"


def main():
    parser = argparse.ArgumentParser(description="blabla.")
    parser.add_argument("--lineage", type=str, required=True, help="Lineage (e.g., Haematopoeitic_lineage, Stromal)")
    parser.add_argument("--use_rep", type=str, required=True, help="Embedding key from obsm (e.g., X_pca, X_scvi)")
    parser.add_argument("--decompose_method", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    args = parser.parse_args()

    # Prepare input_trajectory
    input_trajectories_path_litc = os.path.join(dataset_dir, f"adata_suo_input_haematopoeitic_lineage_litc_{args.decompose_method}.pkl")
    with open(input_trajectories_path_litc, "rb") as _file:
        litc = pickle.load(_file)
        
    little = litc.get_trajectory(args.trajectory, include_additional_nodes=False)
    root_label = little.get_unique_root()

    # Load complete dataset
    adata = sc_suo_developmental_complete(dataset_dir=dataset_dir)

    # Subset to the specified lineage
    adata = adata[adata.obs["LVL0"] == args.lineage]

    # Generate output filename
    lineage_part = args.lineage.replace("_lineage", "").lower()

    output_file = os.path.join(dataset_dir, f"_metric_adata_suo_bootstrap_method_{args.decompose_method}_{lineage_part}_{args.use_rep}_{args.trajectory}.pickle")
    output_file_api = os.path.join(
        dataset_dir, f"_metric_adata_suo_bootstrap_method_{args.decompose_method}_{lineage_part}_{args.use_rep}_{args.trajectory}_api.pickle"
    )

    sc.pp.subsample(adata, n_obs=128000, random_state=0)    
    
    if not args.use_rep.startswith("tardis_"):
        adata = ad.AnnData(X=adata.obsm[args.use_rep].copy(), obs=adata.obs.copy())
    else:
        adata = ad.AnnData(X=adata.obsm[args.use_rep][:, -24:].copy(), obs=adata.obs.copy())

    gc.collect()
    print(adata)

    api = TrajectoryEvaluationAPI(
        adata=adata,  # only hematopoietic
        input_trajectories=little,
        labels_obs="LVL3",
        root_label=root_label,
        logger_level="DEBUG",
    )

    api.evaluate_with_defaults()
    df = api.get_all_results()
    print(df)
    df.to_pickle(output_file)
    print(f"Saved result for {args.lineage!r} with {args.use_rep!r} of group {args.trajectory!r} to {output_file!r} for method {args.decompose_method!r}")


if __name__ == "__main__":
    main()
