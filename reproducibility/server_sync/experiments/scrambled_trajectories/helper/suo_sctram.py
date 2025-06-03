print("started")
import argparse
import gc
import os
import pickle
import sys

import anndata as ad
import numpy as np
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

def smart_subsample(adata, group_key="LVL3", max_cells_per_group=500, random_state=0):
    """
    Subsample overrepresented groups to 'max_cells_per_group' cells,
    retain all cells from underrepresented groups.
    """
    np.random.seed(random_state)
    groups = adata.obs[group_key]
    indices_to_keep = []

    for group in groups.unique():
        group_indices = adata.obs.index[groups == group].tolist()
        if len(group_indices) > max_cells_per_group:
            selected = np.random.choice(group_indices, max_cells_per_group, replace=False)
            indices_to_keep.extend(selected)
        else:
            indices_to_keep.extend(group_indices)

    return adata[indices_to_keep].copy()

def main():
    parser = argparse.ArgumentParser(description="blabla.")
    parser.add_argument("--lineage", type=str, required=True, help="Lineage (e.g., Haematopoeitic_lineage, Stromal)")
    parser.add_argument("--use_rep", type=str, required=True, help="Embedding key from obsm (e.g., X_pca, X_scvi)")
    parser.add_argument("--trajectory_object_path", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    args = parser.parse_args()

    # Prepare input_trajectory
    with open(args.trajectory_object_path, "rb") as _file:
        lit = pickle.load(_file)
    little = lit.get_trajectory(args.trajectory, include_additional_nodes=False)
    root_label = little.get_unique_root()

    # Load complete dataset
    adata = sc_suo_developmental_complete(dataset_dir=dataset_dir)

    # Subset to the specified lineage
    adata = adata[adata.obs["LVL0"] == args.lineage]

    # Generate output filename
    lineage_part = args.lineage.replace("_lineage", "").lower()

    output_file = os.path.join(dataset_dir, f"metric_sctram_suo_scrambling_{lineage_part}_{args.use_rep}_{args.trajectory}.pickle")

    # suo is large, hence make it smaller 
    adata = smart_subsample(adata, group_key="LVL3", max_cells_per_group=1500, random_state=0)
        
    if not args.use_rep.startswith("tardis_"):
        adata = ad.AnnData(X=adata.obsm[args.use_rep].copy(), obs=adata.obs.copy())
    else:
        adata = ad.AnnData(X=adata.obsm[args.use_rep][:, -24:].copy(), obs=adata.obs.copy())

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
    print(f"Saved result for {args.lineage!r} with {args.use_rep!r} of group {args.trajectory!r} to {output_file!r}")


if __name__ == "__main__":
    main()
