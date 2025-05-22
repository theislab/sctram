print("started")

import argparse
import gc
import os
import pickle
import sys
import re

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
    parser.add_argument("--c1", type=str, required=True)
    parser.add_argument("--c2", type=str, required=True)
    parser.add_argument("--c3", type=str, required=True)
    parser.add_argument("--c4", type=int, required=True)
    parser.add_argument("--label_obs_column", type=str, required=True)
    parser.add_argument("--trajectory_path", type=str, required=True)
    args = parser.parse_args()

    print(args)
    print()

    with open(args.trajectory_path, "rb") as _file_trajectory_path:
        trajectory_object = pickle.load(_file_trajectory_path)

    result_dict = dict()
    base_job_name = f"{args.run_id}_c1_{args.c1}_c2_{args.c2}_c4_{args.c4}"
    output_file_path = os.path.join(dataset_dir, f"metric_sctram_{base_job_name}.pkl")
    
    for trj in sorted(trajectory_object.graph["trajectories"]):
    
        latent = ad.read_h5ad(args.c3)
        trajectory = trajectory_object.get_trajectory(trj, include_additional_nodes=False)
        root_label = trajectory.get_unique_root()

        print(trj)
        print(latent)
        print(trajectory)
    
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
        result_dict[trj] = df.copy()
        del latent, trajectory, root_label, api, df
        gc.collect()

    with open(output_file_path, "wb") as _f:
        pickle.dump(result_dict, _f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved result to {output_file_path!r}")

if __name__ == "__main__":
    main()
