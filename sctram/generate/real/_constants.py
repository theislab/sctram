#!/usr/bin/env python3

# Define the DATASETS dictionary with all available datasets
DATASETS = {
    # Drug perturbation:
    "norman_sciplex_cpa": {
        "url": "https://drive.google.com/file/d/1nWZ64abGDFLwhFCaEQUH02f3U5I-t51v/view?usp=share_link",
        "filename": "norman_sciplex_cpa.h5ad",
        "description": "Sciplex dose response dataset from Norman et al.",
    },
    # Developmental:
    "suo_developmental_complete": {
        "url": "https://drive.google.com/file/d/1d_tTFEWoKfL-zPAs6D8KVeQr0dU4NhCU/view?usp=share_link",
        "filename": "suo_developmental_complete.h5ad",
        "description": "Suo Developmental Complete single-cell RNA-seq dataset.",
    },
    "miller_developmental_complete": {
        "url": "https://drive.google.com/file/d/1aOQXRWzkgjchW_WUynlYzkVWWat8qyQW/view?usp=share_link",
        "filename": "miller_developmental_complete.h5ad",
        "description": "Miller Developmental Complete single-cell RNA-seq dataset.",
    },
    "braun_developmental_complete": {
        "url": "https://drive.google.com/file/d/1RmqpDs0LjBlUlbX_SMWmC6I5Zd-KeuJh/view?usp=share_link",
        "filename": "braun_developmental_complete.h5ad",
        "description": "Braun Developmental Complete single-cell RNA-seq dataset.",
    },
    "garcia_developmental_complete": {
        "url": "https://drive.google.com/file/d/1z1u_mc8ou9G_jkK5T3SfJjSd1Vt57H-5/view?usp=share_link",
        "filename": "garcia_developmental_complete.h5ad",
        "description": "Garcia Developmental Complete single-cell RNA-seq dataset.",
    },
    "he_developmental_complete": {
        "url": "https://drive.google.com/file/d/1Yhd-Nt5kGphZz1bYItJSRSAm-JVxm34J/view?usp=share_link",
        "filename": "he_developmental_complete.h5ad",
        "description": "He Developmental Complete single-cell RNA-seq dataset.",
    },
    "kanemaru_developmental_complete": {
        "url": "https://drive.google.com/file/d/1vjOBgzgUGIWqkdfVSiK8I9Wiv_oex3DK/view?usp=share_link",
        "filename": "kanemaru_developmental_complete.h5ad",
        "description": "Kanemaru Developmental Complete single-cell RNA-seq dataset.",
    },
    # scanpy:
    "datasets_krumsiek11": {
        "url": "https://drive.google.com/file/d/1FBcGOe_P1SOxn5Y5dvx8bE7uKOBfenMV/view?usp=share_link",
        "filename": "sc_datasets_krumsiek11.h5ad",
        "description": "The literature-curated boolean network from Krumsiek et al. [2011]. From scanpy package.",
    },
    "datasets_paul15": {
        "url": "https://drive.google.com/file/d/1wIcVzFDKGvl554ctOZmrAtDECNgJsd7D/view?usp=share_link",
        "filename": "sc_datasets_paul15.h5ad",
        "description": "Development of Myeloid Progenitors [Paul et al., 2015].. From scanpy package.",
    },
}
