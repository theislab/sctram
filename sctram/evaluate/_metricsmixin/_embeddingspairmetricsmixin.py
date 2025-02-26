#!/usr/bin/env python3

# TODO: Codebase is not tested and/or runned.
# Note: The codebase here actually belongs to the previous version of the codebase.
# It was kept as reference.

from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase


class EmbeddingsPairMetricsMixin(MetricsMixinBase):
    """Aims to compare two embeddings.

    In the context of single-cell data integration, this embeddings come from anndata before integration and anndata
    after integration.
    """

    available_metrics = [
        # "procrustes",
        # "pearson_correlation",
        # "spearman_correlation",
        # "kendall_correlation",
        # "mean_squared_error",
        # "mean_absolute_error",
        # "r2_score",
        # "cosine_similarity",
        # "embedding_distance",
        # "alignment_score",
        # # "morans_i",
        # # "gearys_c",
        # # "local_morans_i",
        # # "getis_ord_gi_star",
        # "wasserstein_distance",
        # "ks_statistic",
        # "mutual_information",
    ]

    pass
