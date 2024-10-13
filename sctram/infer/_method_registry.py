#!/usr/bin/env python3

from sctram.infer._dpt import DPTInference
from sctram.infer._paga import PAGAInference
from sctram.infer._diffmap import DiffMapInference

method_registry = {
    "paga": PAGAInference,
    "dpt": DPTInference,
    "diffmap": DiffMapInference
    # Add other methods here
}
