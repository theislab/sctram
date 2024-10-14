#!/usr/bin/env python3

from sctram.infer._diffmap import DiffmapEmbedding
from sctram.infer._dpt import DPTInference
from sctram.infer._obsm import ObsmEmbedding
from sctram.infer._paga import PAGAInference
from sctram.infer._pca import PCAEmbedding
from sctram.infer._umap import UMAPEmbedding

method_registry = {
    # Inference Methods
    "paga": PAGAInference,
    "dpt": DPTInference,
    # Embeddings
    "diffmap": DiffmapEmbedding,
    "pca": PCAEmbedding,
    "umap": UMAPEmbedding,
    "obsm": ObsmEmbedding,
}
