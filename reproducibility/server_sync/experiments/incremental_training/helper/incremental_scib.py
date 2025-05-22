print("started")

import argparse
import os
import sys
import gc
import pickle
import h5py

import anndata as ad
import scanpy as sc
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

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
    parser.add_argument("--batch_obs_column", type=str, required=True)
    parser.add_argument("--steps_paths_dict", type=str, required=True)
    parser.add_argument("--latent_step", type=int, required=True)
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    args = parser.parse_args()

    print()
    print("args.run_id", args.run_id)
    print("args.label_obs_column", args.label_obs_column)
    print("args.batch_obs_column", args.batch_obs_column)
    print("args.training_anndata_path", args.training_anndata_path)
    print("args.trajectory_path", args.trajectory_path)
    print("args.steps_paths_dict", args.steps_paths_dict)
    print("args.latent_step", args.latent_step)
    print("args.trajectory", args.trajectory)
    print()

    with open(args.steps_paths_dict, "rb") as _file_pickle_paths:
        pickle_paths = pickle.load(_file_pickle_paths)

    with open(args.trajectory_path, "rb") as _file_trajectory_path:
        trajectory_object = pickle.load(_file_trajectory_path)

    if args.batch_obs_column.lower() == "none":
        batch_obs_column = None
    else:
        batch_obs_column = args.batch_obs_column

    adata = ad.read_h5ad(args.training_anndata_path)
    testing_obsm_name = f"{args.trajectory}_{args.latent_step}"
    adata.obsm[testing_obsm_name] = ad.read_h5ad(pickle_paths[int(args.latent_step)]).X.copy()
    
    print("calculata pca")
    sc.pp.normalize_total(adata, target_sum=1e4)   # CPM / library-size scaling
    sc.pp.log1p(adata)                             # ln(1 + CPM)
    sc.tl.pca(adata, n_comps=20)
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"].copy()
    del adata.uns["pca"]              # eigenvalues, variance explained, …
    del adata.varm["PCs"]             # gene loadings
    del adata.obsm["X_pca"]           # duplicate of what we just saved

    print(adata)
    
    output_file_path = os.path.join(dataset_dir, f"metric_scib_{args.run_id}_step_{args.latent_step}_trajectory_{args.trajectory}.pkl") 

    # load training anndata
    # calculate PCA and make it for `unintegrated`.
    
    if args.trajectory != "all_data_included":
        trajectory = trajectory_object.get_trajectory(args.trajectory, include_additional_nodes=False)
        trajectory_cells = set(trajectory.nodes())
        adata = adata[adata.obs[args.label_obs_column].isin(trajectory_cells)]

    adata = adata.copy()
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

    len_batch = len(adata.obs[batch_obs_column].unique())
    
    if len_batch <= 1:
        batchcor = BatchCorrection(
            silhouette_batch=False,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=False
        ) # there is no batch there. so the tool is raising error.
    else:
        batchcor = BatchCorrection()
    biocons = BioConservation()

    bm = Benchmarker(
        adata=adata,
        batch_key=batch_obs_column,
        label_key=args.label_obs_column,
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

    df.to_pickle(output_file_path)
    print(f"Saved result to {output_file_path!r}")

if __name__ == "__main__":
    main()
