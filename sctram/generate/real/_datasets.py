#!/usr/bin/env python3

from sctram.generate.real._download import download_dataset
import anndata as ad
from scanpy import settings
import scanpy as sc

from sctram.generate.real._urls import (
    url_norman_sciplex_cpa
)

def sc_norman_sciplex_cpa(overwrite=False, timeout=10) -> ad.AnnData:
    """Sciplex dose response dataset from Norman et al.
    
    Modified and simplified version is downloaded from CPA paper. 

    Returns:
        ad.AnnData: A single-cell RNA seq dataset.
    """
    output_file_name = "norman_sciplex_cpa.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    download_dataset(
        url=url_norman_sciplex_cpa,
        filename=output_file_name,
        destination=settings.datasetdir,
        overwrite=overwrite,
        timeout=timeout,
        unzip=False
    )
    
        
    adata = sc.read_h5ad(output_file_path)

    return adata

