print("started")

import argparse
import gc
import os
import pickle
import h5py
import sys

import anndata as ad
import scanpy as sc

# Add working directory to path
working_directory = "/home/icb/kemal.inecik/work/codes/sctram"
sys.path.append(working_directory)

from sctram.api._lower_level import TrajectoryEvaluationAPI
from sctram.input import InputTrajectories

print("libraries loaded")
sc.settings.verbose = 3

dataset_dir = "/home/icb/kemal.inecik/lustre_workspace/temp_sctram_data"
helpers_directory = os.path.join(os.getcwd(), "helper")
logs_directory = os.path.join(os.getcwd(), "logs")

def main():
    parser = argparse.ArgumentParser(description="blabla.")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--training_anndata_path", type=str, required=True)
    parser.add_argument("--label_obs_column", type=str, required=True)
    parser.add_argument("--steps_paths_dict", type=str, required=True)
    parser.add_argument("--latent_step", type=int, required=True)
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    args = parser.parse_args()

    print()
    print("args.run_id", args.run_id)
    print("args.label_obs_column", args.label_obs_column)
    print("args.training_anndata_path", args.training_anndata_path)
    print("args.trajectory_path", args.trajectory_path)
    print("args.steps_paths_dict", args.steps_paths_dict)
    print("args.latent_step", args.latent_step)
    print("args.trajectory", args.trajectory)
    print()

    with h5py.File(args.training_anndata_path) as _file_anndata_training:
        adata_training_obs = ad.io.read_elem(_file_anndata_training["obs"])
    
    with open(args.steps_paths_dict, "rb") as _file_pickle_paths:
        pickle_paths = pickle.load(_file_pickle_paths)

    with open(args.trajectory_path, "rb") as _file_trajectory_path:
        trajectory_object = pickle.load(_file_trajectory_path)
    
    latent = ad.read_h5ad(pickle_paths[int(args.latent_step)])
    latent.obs = adata_training_obs.copy()

    trajectory = trajectory_object.get_trajectory(args.trajectory, include_additional_nodes=False)
    root_label = trajectory.get_unique_root()
    
    print(latent)
    print(trajectory_object)
    print(trajectory)
    
    output_file_path = os.path.join(dataset_dir, f"metric_sctram_{args.run_id}_step_{args.latent_step}_trajectory_{args.trajectory}.pkl") 

    api = TrajectoryEvaluationAPI(
        adata=latent,
        input_trajectory=trajectory,
        labels_obs=args.label_obs_column,
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
