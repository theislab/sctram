#!/usr/bin/env python3

from sctram.generate.real._constants import DATASETS
from sctram.generate.real._downloader import _create_downloader_function

# Dynamically generate downloader functions for each dataset
for key in DATASETS.keys():
    func = _create_downloader_function(key)
    globals()[func.__name__] = func

# Define __all__ for explicit exports
__all__ = [f"sc_{key}" for key in DATASETS.keys()]
