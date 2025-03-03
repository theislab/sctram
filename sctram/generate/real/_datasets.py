#!/usr/bin/env python3

from sctram.generate.real._constants import DATASETS
from sctram.generate.real._downloader import _create_downloader_function

# Dynamically generate downloader functions for each dataset
for key in DATASETS.keys():
    func = _create_downloader_function(key)
    globals()[func.__name__] = func

# Define __all__ for explicit exports
__all__ = [f"sc_{key}" for key in DATASETS.keys()]


# Create a detailed available_datasets report
def available_datasets():
    availables = [
        {"name": key, "description": value["description"], "url": value["url"], "filename": value["filename"]}
        for key, value in DATASETS.items()
    ]
    for dataset in availables:
        print(f"Dataset Name: {dataset['name']}")
        print(f"Description: {dataset['description']}")
        # print(f"URL: {dataset['url']}")
        print(f"Filename: {dataset['filename']}\n")
