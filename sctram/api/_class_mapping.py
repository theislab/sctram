#!/usr/bin/env python3

# Inference Modules

from sctram.infer._dpt import DPTInference
from sctram.infer._paga import PAGAInference
from sctram.infer._pca import PCAEmbedding
from sctram.infer._umap import UMAPEmbedding
from sctram.infer._diffmap import DiffmapEmbedding
from sctram.infer._obsm import ObsmEmbedding

# Evaluation Modules

from sctram.evaluate._pseudotime import PseudotimeValuesEvaluation
from sctram.evaluate._adjacency import AdjacencyMatrixEvaluation
from sctram.evaluate._embedding import EmbeddingTrajectoryEvaluation

# Mapping of Method Names to Classes

PSEUDOTIME_METHODS = {
    "DPTInference": DPTInference,
    # Add other pseudotime inference classes as needed.
}

ADJACENCY_METHODS = {
    "PAGAInference": PAGAInference,
    # Add other adjacency inference classes as needed.
}

EMBEDDING_METHODS = {
    "ObsmEmbedding": ObsmEmbedding,
    "PCAEmbedding": PCAEmbedding,
    "UMAPEmbedding": UMAPEmbedding,
    "DiffmapEmbedding": DiffmapEmbedding,
    # Add any additional embedding methods.
}

# Mapping of Evaluation Classes to Allowed Inference Keys

EVALUATION_METHODS = {
    "PseudotimeValuesEvaluation": PseudotimeValuesEvaluation,
    "AdjacencyMatrixEvaluation": AdjacencyMatrixEvaluation,
    "EmbeddingTrajectoryEvaluation": EmbeddingTrajectoryEvaluation,
}

EVALUATION_ALLOWED_INFER = {
    "PseudotimeValuesEvaluation": list(PSEUDOTIME_METHODS.keys()),
    "AdjacencyMatrixEvaluation": list(ADJACENCY_METHODS.keys()),
    "EmbeddingTrajectoryEvaluation": list(EMBEDDING_METHODS.keys()),
}
