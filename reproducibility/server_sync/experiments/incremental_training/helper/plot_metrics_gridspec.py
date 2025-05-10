import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import numpy as np
import tqdm

def plot_metrics_gridspec(
    df,
    *,
    fig_height_per_metric: int = 6,
    n_col: int = 1,
    x_axis_log_scale: bool = False,
    drop_zero: bool = False,
    trajectory: list | None = None,
    x_range: tuple[float, float] | None = None,
    marker: str = "o",
    markersize: float = 3,
    linewidth: float = 1,
    alpha_line: float = 0.8,
    alpha_marker: float = 0.9,
    second_line_label_suffix: str | None = None, 
):
    # ── Filter data ────────────────────────────────────────────────────────
    if trajectory is not None:
        df = df[df.index.get_level_values("trajectory").isin(trajectory)]

    unique_metrics = df.index.get_level_values("metric").unique().tolist()
    n_metrics = len(unique_metrics)
    n_rows = (n_metrics + n_col - 1) // n_col

    # Unique trajectory‑representation combos (ignore metric + path)
    unique_combos = (
        df.index.droplevel(["metric", "path"]).unique().tolist()
    )
    palette = sns.color_palette(cc.glasbey_category10, len(unique_combos))
    combo_to_color = {c: palette[i] for i, c in enumerate(unique_combos)}

    # ── Figure + GridSpec ──────────────────────────────────────────────────
    fig = plt.figure(
        figsize=(10 * n_col, fig_height_per_metric * n_rows),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(n_rows, n_col)

    for i, metric in enumerate(tqdm.tqdm(unique_metrics, desc="Plotting metrics", ncols=60)):
        row, col = divmod(i, n_col)
        ax = fig.add_subplot(gs[row, col])

        # Pull out single‑metric slice and drop unused levels
        subset_xs = df.xs(metric, level="metric")
        path = subset_xs.index.get_level_values("path").unique().item()
        subset = subset_xs.droplevel("path")

        # Sort epoch columns
        sorted_cols = sorted(subset.columns)
        if drop_zero:
            sorted_cols = [c for c in sorted_cols if c != 0]
        subset = subset[sorted_cols]
        max_epoch = sorted_cols[-1] if sorted_cols else 0

        # X‑axis formatting
        if x_axis_log_scale:
            ax.set_xscale("log")

        if x_range is not None:
            ax.set_xlim(x_range)
        else:
            if not x_axis_log_scale:
                ax.set_xticks(np.arange(0, max_epoch + 1, 50))
                if drop_zero and sorted_cols:
                    ax.set_xlim(left=sorted_cols[0])

        # ── Plot each trajectory‑representation combo ────────────────────
        for combo in unique_combos:
            y = subset.loc[combo].values
            x = subset.loc[combo].index.astype(float)
            color = combo_to_color[combo]

            # Line
            ax.plot(
                x,
                y,
                color=color,
                linewidth=linewidth,
                alpha=alpha_line,
                label=str(combo),
            )

            # Markers
            ax.scatter(
                x,
                y,
                s=markersize**2,
                marker=marker,
                color=color,
                edgecolor="none",
                alpha=alpha_marker,
            )

        # ── Per‑subplot styling ──────────────────────────────────────────
        ax.set_xlabel("Training Epoch", fontsize=11, labelpad=6)
        
        
        # ax.set_ylabel(f"{metric}\n{path}", fontsize=11, labelpad=6)
        
        # Y label with second line in italic
        second_line = f"\n$\\it{{{path}}}$ {second_line_label_suffix if second_line_label_suffix else ''}"  
        ax.set_ylabel(f"{metric}{second_line}", fontsize=11, labelpad=6)

        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        sns.despine(ax=ax, left=True, trim=True)

        ax.tick_params(axis="x", rotation=90, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

        # Legend
        ax.legend(
            loc="upper left",
            fontsize=8,
            frameon=True,
            framealpha=0.85,
            ncol=2 if trajectory is None or len(trajectory) > 1 else 1,
        )

    if trajectory is not None:
        fig.suptitle(" | ".join(trajectory), fontsize=14, y=1.01)

    plt.show()
