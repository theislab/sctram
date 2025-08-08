print("started")
import argparse
import gc
import os
import pickle
import sys
import numpy as np

import anndata as ad
import scanpy as sc

print("custom libraries")
# Add working directory to path
sys.path.append("/home/icb/kemal.inecik/work/codes/sctram")
sys.path.append("/home/icb/kemal.inecik/work/codes/sctram/reproducibility/server_sync/experiments")
print(sys.path)

from sctram.api._lower_level import TrajectoryEvaluationAPI
from sctram.generate.real import sc_suo_developmental_complete
from sctram.input import InputTrajectories
from development_atlas_figure.helper.pickle_operations import save_pickle_bundle, load_pickle_bundle

print("libraries loaded")

sc.settings.verbose = 3
dataset_dir = "/home/icb/kemal.inecik/lustre_workspace/temp_sctram_data"


def main():
    parser = argparse.ArgumentParser(description="sctram runs for developmental atlas figure.")
    parser.add_argument("--trajectory_annotation_column", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--use_rep", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()

    print()
    print("args.trajectory_annotation_column", args.trajectory_annotation_column)
    print("args.trajectory", args.trajectory)
    print("args.use_rep", args.use_rep)
    print("args.run_id", args.run_id)
    print()

    pickle_bundle_path = os.path.join(dataset_dir, "_experiment_development_atlas_helper_pickle_bundle.pkl")
    (
        df_hdca_component_adata_paths,
        df_hdca_lvl3_adata_paths,
        out_df,
        ground_truth_trajectories,
        ground_truth_trajectories_annotation_level,
        trajectory_subset_dict
    ) = load_pickle_bundle(pickle_bundle_path)

    m1, m2, m3 = trajectory_subset_dict[args.trajectory]

    # Prepare input_trajectory
    little = InputTrajectories(ground_truth_trajectories).get_trajectory(args.trajectory, include_additional_nodes=False)
    root_label = little.get_unique_root()

    # Load complete dataset
    latent_path = df_hdca_lvl3_adata_paths[args.use_rep]["latent"]
    adata = ad.read_h5ad(latent_path)
    assert np.all(adata.obs.index == out_df.index) or adata.obs.shape[1] == 0
    adata.obs = out_df.copy()
    print(adata)
    
    adata = adata[adata.obs[m1] == m2].copy()
    print(adata)
    sc.pp.subsample(adata, fraction=m3)
    print(adata)

    base_job_name = f"{args.run_id}_c1_{args.trajectory}_c2_{args.use_rep}"
    output_file_path = os.path.join(dataset_dir, f"metric_sctram_{base_job_name}.pkl")
    
    n_trajectory_cells = adata.obs[args.trajectory_annotation_column].isin(set(little.nodes())).sum()
    n_max_trajectory_cells = 80000
    n_ratio = n_max_trajectory_cells/n_trajectory_cells
    if n_ratio<1:
        sc.pp.subsample(adata, fraction=n_ratio, random_state=0)
    adata = adata.copy()

    gc.collect()
    print(adata)

    api = TrajectoryEvaluationAPI(
        adata=adata, 
        input_trajectory=little,
        labels_obs=args.trajectory_annotation_column,
        root_label=root_label,
        logger_level="DEBUG",
    )

    api.evaluate_with_defaults()
    df = api.get_all_results()
    print(df)
    df.to_pickle(output_file_path)
    print(f"Saved result to {output_file_path!r}")


if __name__ == "__main__":
    main()
