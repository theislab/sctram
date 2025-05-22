import numpy as np
import anndata as ad

def remove_random_fraction_from_labels(
    adata: ad.AnnData,
    k: float,
    labels_to_affect: list,
    seed: int = 42,
    cluster_col: str = "clusters"
) -> ad.AnnData:
    """
    Removes a random fraction k (0 < k < 1) of cells from specified cluster labels.

    Parameters:
    - adata: AnnData object
    - k: Fraction of cells to remove per affected label
    - labels_to_affect: List of cluster labels to apply removal to
    - seed: Random seed for reproducibility
    - cluster_col: Column in adata.obs to group by

    Returns:
    - Filtered AnnData object
    """
    rng = np.random.default_rng(seed)
    keep_indices = []

    for label in adata.obs[cluster_col].unique():
        label_indices = adata.obs[adata.obs[cluster_col] == label].index

        if label in labels_to_affect:
            n_remove = int(len(label_indices) * k)
            remove_indices = rng.choice(label_indices, size=n_remove, replace=False)
            keep = set(label_indices) - set(remove_indices)
        else:
            keep = label_indices

        keep_indices.extend(keep)

    return adata[keep_indices].copy()


def remove_random_fraction_of_genes(
    adata: ad.AnnData,
    k: float,
    seed: int = 42
) -> ad.AnnData:
    """
    Removes a random fraction k (0 < k < 1) of genes from the AnnData object,
    preserving the original gene order.

    Parameters:
    - adata: AnnData object
    - k: Fraction of genes to remove (between 0 and 1)
    - seed: Random seed for reproducibility

    Returns:
    - Filtered AnnData object with a subset of genes in original order
    """
    rng = np.random.default_rng(seed)
    n_genes = adata.n_vars
    n_remove = int(n_genes * k)

    remove_indices = rng.choice(n_genes, size=n_remove, replace=False)
    mask = np.ones(n_genes, dtype=bool)
    mask[remove_indices] = False

    return adata[:, mask].copy()
