import gc
import os
import numpy as np
import pandas as pd
import math
from typing import Mapping, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from adjustText import adjust_text
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
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

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


def annotate_points_on_line(
    ax: plt.Axes,
    x_points: list[float],
    *,
    label_fmt: str = "{:.2f}",
    offset_frac: float | None = 0.04,   # ← choose ONE of these …
    offset_abs : float | None = None,   # ← … leave the other as None
    scatter_kwargs: dict | None = None,
    text_kwargs   : dict | None = None,
    arrowprops    : dict | None = None,
):
    """
    Place labels a user-controlled distance to the RIGHT of each marker,
    then draw arrows that touch both label and marker.
    """

    if not ax.lines:
        raise ValueError("No line was found on the supplied Axes.")

    # ——— get line data ——————————————————————————
    line        = ax.lines[0]
    x_data, y_data = map(np.asarray, (line.get_xdata(), line.get_ydata()))
    ord_idx     = np.argsort(x_data)
    x_data, y_data = x_data[ord_idx], y_data[ord_idx]

    # ——— y at requested x ——————————————————————
    if ax.get_xscale() == "log":
        y_points = np.interp(np.log10(x_points), np.log10(x_data), y_data)
    else:
        y_points = np.interp(x_points, x_data, y_data)

    # ——— markers ——————————————————————————————
    scatter_defaults = dict(s=40, facecolor="white", edgecolor="black", zorder=4)
    scatter_defaults.update(scatter_kwargs or {})
    ax.scatter(x_points, y_points, **scatter_defaults)

    # ——— figure out the horizontal offset ————
    if ax.get_xscale() == "log":
        # multiplicative offset: x_out = x_in * 10**d
        if offset_abs is not None:          # absolute multiplier
            x_text = [xp * offset_abs       for xp in x_points]
        else:                               # fractional multiplier
            log_span = np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])
            factor   = 10 ** ( (offset_frac or 0) * log_span )
            x_text = [xp * factor           for xp in x_points]
    else:
        # additive offset: x_out = x_in + d
        if offset_abs is not None:
            dx = offset_abs
        else:
            dx = (offset_frac or 0) * (ax.get_xlim()[1] - ax.get_xlim()[0])
        x_text = [xp + dx                   for xp in x_points]

    # ——— labels ——————————————————————————————
    if text_kwargs is False:
        return []
    
    texts = []
    text_defaults = dict(
        ha="left", va="center", fontsize=8, zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.4, alpha=0.85)
    )
    text_defaults.update(text_kwargs or {})

    for xt, xp, yp in zip(x_text, x_points, y_points):
        txt = ax.text(xt, yp, label_fmt.format(xp), **text_defaults)
        texts.append(txt)

    # ——— arrow & collision-avoidance ——————————
    arrow_defaults = dict(arrowstyle="->", color="black",
                          lw=0.7, mutation_scale=10, shrinkA=0, shrinkB=0)
    arrow_defaults.update(arrowprops or {})

    adjust_text(
        texts,
        x=x_points,
        y=y_points,
        ax=ax,
        expand_points=(1.15, 1.25),
        arrowprops=arrow_defaults
    )

    return texts

    
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
