#!/usr/bin/env python3

from sctram.infer._paga import PAGAInference
from sctram.infer._dpt import DPTInference

method_registry = {
    'paga': PAGAInference,
    'dpt': DPTInference,
    # Add other methods here
}
