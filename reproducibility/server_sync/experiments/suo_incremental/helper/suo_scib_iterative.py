print("started")
import argparse
import os
import sys
import gc
import pickle

import anndata as ad
import scanpy as sc
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# Add working directory to path
working_directory = "/home/icb/kemal.inecik/work/codes/sctram"
sys.path.append(working_directory)
from sctram.generate.real import sc_suo_developmental_complete
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

    adata_suo_complete = sc_suo_developmental_complete(dataset_dir=dataset_dir)
    print(adata_suo_complete)

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

    output_file = os.path.join(dataset_dir, f"_metric_adata_suo_iterative_{lineage_part}_{args.use_rep}_{args.epoch}_{args.itpn_base}_{args.trajectory}_scib.pickle")
    
    n_trajectory_cells = adata.obs["LVL3"].isin(set(little.nodes())).sum()
    n_max_trajectory_cells = 80000
    n_ratio = n_max_trajectory_cells/n_trajectory_cells
    if n_ratio<1:
        sc.pp.subsample(adata, fraction=n_ratio, random_state=0)    
    
    if args.use_rep.startswith("tardis_"):
        raise ValueError

    adata_suo_complete = sc_suo_developmental_complete(dataset_dir=dataset_dir)
    adata.obsm['Unintegrated'] = adata_suo_complete[adata.obs.index].obsm['Unintegrated'].copy()
    testing_obsm_name = f"{args.use_rep}_{args.epoch}"
    adata.obsm[testing_obsm_name] = adata.X.copy()
    del adata_suo_complete
    gc.collect()
    print(adata)

    def get_results(self, min_max_scale: bool = True) -> pd.DataFrame:
        _LABELS = "labels"
        _BATCH = "batch"
        _X_PRE = "X_pre"
        _METRIC_TYPE = "Metric Type"
        _AGGREGATE_SCORE = "Aggregate score"
        
        df = self._results.transpose()
        df.index.name = "Embedding"
        df = df.loc[df.index != _METRIC_TYPE]
        if min_max_scale:
            # Use sklearn to min max scale
            df = pd.DataFrame(
                MinMaxScaler().fit_transform(df),
                columns=df.columns,
                index=df.index,
            )
        df = df.transpose()
        df[_METRIC_TYPE] = self._results[_METRIC_TYPE].values
    
        # Compute scores
        per_class_score = df.groupby(_METRIC_TYPE).mean().transpose()
        # This is the default scIB weighting from the manuscript
        # per_class_score["Total"] = 0.4 * per_class_score["Batch correction"] + 0.6 * per_class_score["Bio conservation"]
        df = pd.concat([df.transpose(), per_class_score], axis=1)
        df.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
        return df

    print("starting scvi scib")
    
    batchcor = BatchCorrection()
    biocons = BioConservation()

    bm = Benchmarker(
        adata=adata,
        batch_key="concatenated_integration_covariates",
        label_key="cell_type",
        embedding_obsm_keys=[testing_obsm_name],
        pre_integrated_embedding_obsm_key="Unintegrated",
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcor,
        n_jobs=-1,
    )

    bm.benchmark()
    print("getting results.")
    df = get_results(bm, min_max_scale=False)
    print(df)

    df.to_pickle(output_file)
    print(f"Saved result to {output_file!r}")


if __name__ == "__main__":
    main()
