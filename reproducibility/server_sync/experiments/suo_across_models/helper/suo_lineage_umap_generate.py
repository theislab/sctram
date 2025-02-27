import os
import sys
import argparse
import scanpy as sc
import anndata as ad

sc.settings.verbose = 3

# Add working directory to path
working_directory = "/home/icb/kemal.inecik/work/codes/sctram"
sys.path.append(working_directory)

dataset_dir = "/home/icb/kemal.inecik/lustre_workspace/temp_sctram_data"

from sctram.generate.real import sc_suo_developmental_complete

def main():
    parser = argparse.ArgumentParser(description='Generate UMAP embeddings for specified lineage and embedding.')
    parser.add_argument('--lineage', type=str, required=True, help='Lineage (e.g., Haematopoeitic_lineage, Stromal)')
    parser.add_argument('--use_rep', type=str, required=True, help='Embedding key from obsm (e.g., X_pca, X_scvi)')
    parser.add_argument('--n_neighbors', type=int, required=True, help='Neighbors for the KNN calculation')
    args = parser.parse_args()

    # Load complete dataset
    adata_suo_complete = sc_suo_developmental_complete(dataset_dir=dataset_dir)

    # Subset to the specified lineage
    adata_lineage = adata_suo_complete[adata_suo_complete.obs["LVL0"] == args.lineage]
    del adata_lineage.uns

    # Check if use_rep exists
    if args.use_rep not in adata_lineage.obsm:
        raise ValueError(f"Embedding {args.use_rep} not found in obsm for lineage {args.lineage}")

    # Generate output filename
    lineage_part = args.lineage.replace('_lineage', '').lower()
    output_file = os.path.join(dataset_dir, f"adata_suo_{lineage_part}_umap_{args.use_rep}.h5ad")

    if not args.use_rep.startswith("tardis_"):
        adata = ad.AnnData(X=adata_lineage.obsm[args.use_rep].copy(), obs=adata_lineage.obs.copy())
    else:
        adata = ad.AnnData(X=adata_lineage.obsm[args.use_rep][:,-24:].copy(), obs=adata_lineage.obs.copy())

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata, n_neighbors=int(args.n_neighbors))
    sc.tl.umap(adata)
    
    # Save results
    adata.write_h5ad(output_file)
    print(f"Saved UMAP for {args.lineage} with {args.use_rep} to {output_file}")

if __name__ == "__main__":
    main()
