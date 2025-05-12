import gc
import os
import pandas as pd
import math
from typing import Mapping, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator, MaxNLocator

def normalize_and_align(df, metric_direction):
    # 1) Z-score per metric
    df["z_score"] = df.groupby("metric")["score"] \
                      .transform(lambda x: (x - x.mean()) / x.std())
    
    # 2) Flip sign for decreasing metrics
    if metric_direction == "None":
        df["aligned_score"] = df["z_score"].copy()
    else:
        df["aligned_score"] = df.apply(
            lambda row: row["z_score"] * (-1 if metric_direction[row["metric"]] == "decreasing" else 1),
            axis=1
        )
    return df


def plot_summary_ci(
    df: pd.DataFrame,
    *,
    x: str = "weight_coef",
    x_label: str = "TarDis Loss Weight",
    y: str = "aligned_score",
    hue: str | None = None,
    hue_order = None,
    ci: float = 0.95,
    log_x: bool = True,
    figsize: tuple[float, float] | None = None,
    color: str = "black",
    line_width: float = 0.81,
    marker=False,                 
    markersize=4,   
    legend_title=None
):
    fig, ax = plt.subplots(figsize=figsize or (8, 3))

    bp = sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue_order=hue_order,
        hue=hue,                # None → single curve
        estimator="mean",
        errorbar=("ci", ci * 100) if ci!=0 else None,
        err_style="band",
        err_kws={
            "alpha": 0.25,
            "edgecolor": "none"
        },
        color=None if hue else color,
        palette=None if not hue else color,
        linewidth=line_width,
        ax=ax,
        marker=marker,       
        markersize=markersize,
        markeredgewidth=0,
    )

    # Edge styling for CI band is handled by seaborn; we just keep grid & despine.
    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label, fontsize=8, labelpad=10)
    
    # ——— polished y-axis ———
    ax.set_ylabel(f"Performance\n(z-score, {int(ci*100)} % CI)", fontsize=8, labelpad=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    # Dashed horizontal grid – subtle
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine(left=True, trim=False)

    # If there is a legend (i.e. `hue` not None), style it like before
    if hue:
        # remove seaborn’s default legend first
        if ax.legend_ is not None:
            ax.legend_.remove()
    
        handles, labels = ax.get_legend_handles_labels()
    
        slim_handles = []
        for h in handles:
            # Line2D → use get_color(); Patch → get_facecolor()
            try:
                color = h.get_facecolor()
                # reduce (N,4) RGBA array to a single RGBA tuple if needed
                if hasattr(color, "__iter__") and len(color) and not isinstance(color, tuple):
                    color = tuple(color[0])
            except AttributeError:
                color = h.get_color()
    
            slim_handles.append(
                mpatches.Patch(
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5
                )
            )
    
        ax.legend(
            slim_handles,
            labels,
            title=hue.replace("_", " ").title() if not legend_title else legend_title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            alignment="left",
            title_fontsize=8,
            fontsize=8,
            borderaxespad=0.5,
            handletextpad=0.4,
            labelspacing=0.5,
        )

    plt.tight_layout(rect=[0, 0, 0.85, 1])   # leave room for legend
    return fig, ax


def plot_umap_grid(
    definitions: Mapping[str, Any],
    dataset_key: str = "tardis_sciplex_dose",
    color: str = "dose",
    n_cols: int = 8,
    figsize_per_plot: float = 2.5,
    reverse: bool = True,
    verbose_missing: bool = True,
):
    """
    Draw a grid of UMAP plots for every *existing* latent embedding in
    definitions[dataset_key]["latent_paths"].

    Parameters
    ----------
    definitions : dict
        Dictionary that holds latent_paths under ``definitions[dataset_key]``.
    dataset_key : str, optional
        Entry in ``definitions`` to use (default ``'tardis_sciplex_dose'``).
    n_cols : int, optional
        Number of columns in the grid (default 8).
    color : str, optional
        .obs column to colour points by (default 'dose').
    figsize_per_plot : float, optional
        Inches per subplot (default 2.5).
    reverse : bool, optional
        Plot keys in descending order (default True).
    verbose_missing : bool, optional
        Print which keys were skipped because their file is missing (default True).
    """
    import anndata as ad
    import scanpy as sc
    import matplotlib.pyplot as plt

    # ---- gather & check -----------------------------------------------------
    all_items = definitions[dataset_key]["latent_paths"].items()
    present = [(k, p) for k, p in all_items if os.path.exists(p)]
    missing = [k for k, p in all_items if not os.path.exists(p)]

    if verbose_missing and missing:
        print(
            f"[plot_umap_grid] Skipping {len(missing)} missing file(s): "
            + ", ".join(map(str, missing))
        )

    if not present:
        raise FileNotFoundError(
            "No latent files found – nothing to plot."
        )

    # ---- order & grid size --------------------------------------------------
    keys, paths = zip(*sorted(present, key=lambda kp: kp[0], reverse=reverse))
    n_items = len(keys)
    n_rows = math.ceil(n_items / n_cols)

    # ---- canvas -------------------------------------------------------------
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_plot, n_rows * figsize_per_plot),
        squeeze=False,
    )

    # ---- iterate ------------------------------------------------------------
    for i, (k, path) in enumerate(zip(keys, paths)):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        adata = ad.read_h5ad(path)

        # compute neighbours / UMAP if needed
        if "X_umap" not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)

        sc.pl.umap(
            adata,
            color=color,
            legend_loc=None,
            frameon=False,
            ax=ax,
            show=False,
        )
        ax.set_title(f"{k:g}", fontsize=8)

        del adata
        gc.collect()

    # ---- blank spare pads ---------------------------------------------------
    for j in range(n_items, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()
