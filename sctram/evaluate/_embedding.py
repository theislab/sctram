#!/usr/bin/env python3

# TODO: Codebase is not tested and/or runned.

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import procrustes
from sklearn.manifold import Isomap

from sctram.evaluate._base import EvaluationBase
from sctram.evaluate._metricsmixin._embeddingspairmetricsmixin import EmbeddingsPairMetricsMixin
from sctram.evaluate._metricsmixin._embeddingtrajectorymetricsmixin import EmbeddingTrajectoryMetricsMixin


class EmbeddingsPairEvaluation(EmbeddingsPairMetricsMixin, EvaluationBase):
    """Evaluation method to compare inferred embeddings with another embedding.

     This class compares the one embedding (numpy array of shape [n_samples, n_components])
    with another embedding, using data point labels, by means of various metrics to assess how well
    the embedding captures the trajectory structure.
    """

    pass  # TODO: complete the class.


class EmbeddingTrajectoryEvaluation(EmbeddingTrajectoryMetricsMixin, EvaluationBase):
    """Evaluation method to compare inferred embeddings with a given trajectory.

    This class compares the inferred embeddings (numpy array of shape [n_samples, n_components])
    with the given trajectory (networkx.MultiDiGraph), using various metrics to assess how well
    the embedding captures the trajectory structure.
    """

    pass  # TODO: complete the class.
