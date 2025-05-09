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
    parser = argparse.ArgumentParser(description="sctram runs for developmental atlas figure.")
    parser.add_argument("--latent_path", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    args = parser.parse_args()

    print()
    print("args.latent_path", args.latent_path)
    print("args.trajectory_path", args.trajectory_path)
    print("args.trajectory", args.trajectory)
    print()

    # Prepare input_trajectory
    with open(args.trajectory_path, "rb") as _file:
        litc = pickle.load(_file)
    little = litc.get_trajectory(args.trajectory, include_additional_nodes=False)
    root_label = little.get_unique_root()

    # Load complete dataset
    adata = ad.read_h5ad(args.latent_path)

    base_name = os.path.splitext(os.path.basename(args.latent_path))[0]
    output_file = os.path.join(dataset_dir, f"_metric_{base_name}_{args.trajectory}.pickle")
    
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
