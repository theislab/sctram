import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from adjustText import adjust_text  
import matplotlib.patheffects as path_effects


def plot_obsms(lin, plot_obs, adata, label_on_data=False, n_cols=4, subplot_size=(5, 4), prevent_bottom_legend=False, coefficient=1.0):
    # Collect all categories and their global counts for consistent coloring
    lin = lin.replace('_lineage', '').lower()
    global_counts = {}
    
    # First pass to get all categories and their global counts
    for use_rep in adata[lin]:
        adata_obj = adata[lin][use_rep]
        counts = adata_obj.obs[plot_obs].astype(str).value_counts().to_dict()
        for cat, count in counts.items():
            global_counts[cat] = global_counts.get(cat, 0) + count
        break  # as all the reps have the same anndata.obs
    
    # Sort categories by global abundance descending
    unique_lvl = sorted(global_counts.keys(), key=lambda x: -global_counts[x])
    
    # Create color mapping based on global abundance
    n_cats = len(unique_lvl)
    palette = sns.color_palette(cc.glasbey_category10, n_cats)
    lvl_to_color = {cat: palette[i] for i, cat in enumerate(unique_lvl)}
    
    # Plotting parameters
    n_cols = n_cols
    subplot_size = subplot_size
    legend_ncol = 4
    
    use_reps = list(adata[lin].keys())
    if not use_reps:
        raise ValueError(f"no obsm for {lin!r}")
        
    n_subplots = len(use_reps)
    n_rows = int(np.ceil(n_subplots / n_cols))
    n_datapoints = len(next(iter(adata[lin].values())))  # Get count from first adata_obj
    
    # Dynamic point size calculation
    base_size = 1e5
    s = base_size / n_datapoints * coefficient
    
    # Figure setup
    fig_width = subplot_size[0] * n_cols
    fig_height = subplot_size[1] * n_rows + 2
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    
    # Plot each use_rep with local abundance ordering
    for idx, use_rep in enumerate(use_reps):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        adata_obj = adata[lin][use_rep]
        umap_coords = adata_obj.obsm['X_umap']
        
        # Get local counts for layering order
        local_counts = adata_obj.obs[plot_obs].astype(str).value_counts().sort_values(ascending=False)
        sorted_cats = local_counts.index.tolist()
        
        # Plot each category in abundance order (large to small)
        for cat in sorted_cats:
            mask = adata_obj.obs[plot_obs].astype(str) == cat
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                s=s,
                c=[lvl_to_color[cat]],
                alpha=0.4,
                linewidth=s*0.3,
                # rasterized=True,
                edgecolor='none'
            )
        
        # Add labels on data if enabled
        if label_on_data:
            texts = []
            centroid_info = []  # Stores (centroid, color) for each label
            
            for cat in sorted_cats:
                mask = adata_obj.obs[plot_obs].astype(str) == cat
                coords = umap_coords[mask]
                if len(coords) == 0:
                    continue
                
                # Compute centroid with outlier removal
                initial_centroid = np.mean(coords, axis=0)
                distances = np.linalg.norm(coords - initial_centroid, axis=1)
                std = np.std(distances)
                if std == 0:
                    final_centroid = initial_centroid
                else:
                    threshold = 3.0 * std
                    filtered = distances <= threshold
                    filtered_coords = coords[filtered]
                    final_centroid = np.mean(filtered_coords, axis=0) if len(filtered_coords) > 0 else initial_centroid
                
                # Plot enlarged centroid point
                ax.scatter(
                    final_centroid[0], final_centroid[1],
                    s=30*s,  # 10x normal size
                    c=[lvl_to_color[cat]],
                    alpha=1.0,
                    edgecolor='white',
                    linewidth=s*1,
                    zorder=5
                )
                
                # Create squeezed text using LaTeX \scalebox
                txt = ax.text(
                    final_centroid[0], final_centroid[1],
                    cat,
                    fontsize=6,
                    weight='bold',  # Make text bold
                    ha='center',
                    va='center',
                    color="black", #lvl_to_color[cat],
                    zorder=6
                )
                # Add a white outline effect around the text
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=0.5, foreground='white'),
                    path_effects.Normal()
                ])
                
                texts.append(txt)
                centroid_info.append((final_centroid, lvl_to_color[cat]))
            
            # Adjust text positions without automatic arrows
            if texts:
                adjust_text(
                    texts, 
                    ax=ax,
                    force_text=(0.3, 0.3),   # Increase these values for stronger text movement
                    force_points=(1.8, 1.8),
                    lim=500
                )
                
                # Add manual colored arrows
                for txt, (centroid, color) in zip(texts, centroid_info):
                    # Get the text bounding box in data coordinates
                    renderer = fig.canvas.get_renderer()
                    bbox = txt.get_window_extent(renderer=renderer)
                    bbox_data = ax.transData.inverted().transform(bbox)
                    
                    # Calculate the center of the text bounding box
                    text_center = [
                        (bbox_data[0, 0] + bbox_data[1, 0]) / 2,  # x-center
                        (bbox_data[0, 1] + bbox_data[1, 1]) / 2   # y-center
                    ]
                    
                    # Draw arrow from centroid to text center
                    ax.annotate(
                        '',
                        xy=centroid,  # Arrow points to text center
                        xytext=text_center,  # Arrow starts at centroid
                        arrowprops=dict(
                            arrowstyle='->',
                            color="black", #color,
                            lw=0.5,
                            alpha=1,
                            connectionstyle="arc3,rad=0.15"  # Adjust curvature as needed
                        ),
                        zorder=4
                    )
        
        # Subplot formatting
        ax.set_title(use_rep, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    if not label_on_data and not prevent_bottom_legend:
        # Create unified legend only if labels are not on data
        legend_handles = [
            plt.Line2D([0], [0], 
             marker='o', 
             color='w', 
             markerfacecolor=lvl_to_color[cat], 
             markersize=10,
             label=f"{cat} ({global_counts[cat]:,})")
            for cat in unique_lvl
        ]
        
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=legend_ncol,
            bbox_to_anchor=(0.5, -0.05),
            frameon=False,
            fontsize=10,
            handletextpad=0.1
        )

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.suptitle(f"{lin} - {plot_obs}", y=0.98, fontsize=12)
    plt.show()
