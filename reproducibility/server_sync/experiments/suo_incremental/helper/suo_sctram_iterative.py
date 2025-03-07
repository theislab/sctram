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
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--itpn_base", type=str, required=True)
    args = parser.parse_args()

    print()
    print("args.lineage", args.lineage)
    print("args.use_rep", args.use_rep)
    print("args.epoch", args.epoch)
    print("args.trajectory", args.trajectory)
    print("args.itpn_base", args.itpn_base)
    print()

    # Prepare input_trajectory
    input_trajectories_path_name = args.itpn_base + ".pkl"
    input_trajectories_path_litc = os.path.join(dataset_dir, input_trajectories_path_name)
    with open(input_trajectories_path_litc, "rb") as _file:
        litc = pickle.load(_file)
    little = litc.get_trajectory(args.trajectory, include_additional_nodes=False)
    root_label = little.get_unique_root()

    # Load complete dataset
    adata_path = os.path.join(dataset_dir, f"adata_suo_incremental_training_{args.use_rep}_epoch_{args.epoch}.h5ad")
    adata = ad.read_h5ad(adata_path)

    # Subset to the specified lineage
    adata = adata[adata.obs["LVL0"] == args.lineage]

    # Generate output filename
    lineage_part = args.lineage.replace("_lineage", "").lower()

    output_file = os.path.join(dataset_dir, f"_metric_adata_suo_iterative_{lineage_part}_{args.use_rep}_{args.epoch}_{args.itpn_base}_{args.trajectory}.pickle")
    
    n_trajectory_cells = adata.obs["LVL3"].isin(set(little.nodes())).sum()
    n_max_trajectory_cells = 80000
    n_ratio = n_max_trajectory_cells/n_trajectory_cells
    if n_ratio<1:
        sc.pp.subsample(adata, fraction=n_ratio, random_state=0)    
    
    if args.use_rep.startswith("tardis_"):
        raise ValueError

    gc.collect()
    print(adata)

    api = TrajectoryEvaluationAPI(
        adata=adata,  # only hematopoietic
        input_trajectory=little,
        labels_obs="LVL3",
        root_label=root_label,
        logger_level="DEBUG",
    )

    api.evaluate_with_defaults()
    df = api.get_all_results()
    print(df)
    df.to_pickle(output_file)
    print(f"Saved result to {output_file!r}")


if __name__ == "__main__":
    main()
