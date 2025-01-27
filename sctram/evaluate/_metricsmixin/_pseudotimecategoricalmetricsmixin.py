#!/usr/bin/env python3

# TODO: Codebase is not tested and/or runned.

from sctram.evaluate._metricsmixin._metricsmixinbase import MetricsMixinBase
from sctram.evaluate._metricsmixin._spatialmetricsmixin import SpatialMetricsMixin


class PseudotimeCategoricalMetricsMixin(MetricsMixinBase, SpatialMetricsMixin):
    """Pseudotime metrics with categorical labels."""

    pass
