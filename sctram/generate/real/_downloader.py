#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import anndata as ad
import scanpy as sc
from scanpy import settings

from sctram.generate.real._constants import DATASETS
from sctram.generate.real._download import download_dataset
from sctram.utils._loguru_scanpy_capture import redirect_scanpy_logs_to_loguru


def _inherit_docstring(source):
    """A decorator function that copies the docstring from the source function to the target function.

    Args:
        source (function): The function from which the docstring will be inherited.

    Returns:
        function: A decorator that transfers the docstring from the source function
        to the target function.
    """

    def decorator(target):
        target.__doc__ = source.__doc__
        return target

    return decorator


class _DatasetDownloader:
    """Downloader for Google Drive-hosted single-cell RNA-seq datasets."""

    def __init__(self, dataset_key: str):
        """Initializes the downloader with the specified dataset key.

        Args:
            dataset_key (str): The key identifying the dataset in the registry.

        Raises:
            ValueError: If the dataset_key is not found in the registry.
        """
        if dataset_key not in DATASETS:
            raise ValueError(f"Dataset key {dataset_key!r} not found in the registry.")

        self.dataset_info = DATASETS[dataset_key]
        self.url = self.dataset_info["url"]
        self.filename = self.dataset_info["filename"]
        self.description = self.dataset_info.get("description", "")

    def download_and_load(self, overwrite: bool, timeout: int, dataset_dir: Optional[str]) -> ad.AnnData:
        """Downloads the dataset and loads it into an AnnData object.

        Args:
            overwrite (bool): Whether to overwrite the existing file.
            timeout (int): Timeout for the download in seconds.
            dataset_dir (str, optional): Directory to save the dataset. Defaults to scanpy settings.

        Returns:
            ad.AnnData: The loaded single-cell RNA-seq dataset.
        """
        if dataset_dir is None:
            _dataset_dir = settings.datasetdir
        else:
            _dataset_dir = Path(dataset_dir)

        output_file_path = _dataset_dir / self.filename

        # Download the dataset
        download_dataset(
            url=self.url,
            filename=self.filename,
            destination=_dataset_dir,
            overwrite=overwrite,
            timeout=timeout,
            unzip=False,
        )

        with redirect_scanpy_logs_to_loguru(custom_caller="sc.read_h5ad"):
            return sc.read_h5ad(output_file_path)


# Function factory to create downloader functions
def _create_downloader_function(dataset_key: str):
    """Creates a downloader function for a specific dataset.

    Args:
        dataset_key (str): The key identifying the dataset in the registry.

    Returns:
        function: The downloader function.
    """

    @_inherit_docstring(_DatasetDownloader.download_and_load)
    def downloader(  # noqa: D103
        overwrite: bool = False, timeout: int = 10, dataset_dir: Optional[str] = None
    ) -> ad.AnnData:
        return _DatasetDownloader(dataset_key).download_and_load(
            overwrite=overwrite, timeout=timeout, dataset_dir=dataset_dir
        )

    downloader.__name__ = f"sc_{dataset_key}"
    downloader.__qualname__ = f"sc_{dataset_key}"
    return downloader
