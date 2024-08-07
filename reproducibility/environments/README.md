# scTRAM Reproducibility Environments

Follow the steps.

1. Clone the repo.

    ```bash
    git clone https://github.com/theislab/sctram
    ```

2. Necessary to run beforehand for MAC Silicon:

    > Note: Assumes that [homebrew](https://brew.sh) was already installed.

    ```bash
    brew install openblas
    brew install pyenv --head
    # Make sure package manager can find these paths. Export paths before the installation.
    export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"
    export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"
    ```

    Also:

    ```bash
    brew install pandoc
    ```

3. Install the conda development environments.

    > Note: [mamba](https://github.com/conda-forge/miniforge) (mambaforge) improves install times drastically.
    > All mamba commands can be replaced by `conda`.

    ```bash
    mamba env create -f reproducibility/environments/sctram_poetry_3_9_env.yaml
    # TODO: extend the environments for python 3.10, 3.11, 3.12
    ```

    To remove an environment, run the following:

    ```bash
    mamba env remove -n sctram_poetry_3_9_env; mamba clean -avvvy
    ```

    If you use an Apple Silicon processor, you have to be sure that you have the HDF5 installation already. That
    is why HDF5 library is added to the conda environment.

4. Install the poetry environments.

    Have a look at this [website](https://python-poetry.org/docs/managing-environments/) for further info for `poetry`.

    ```bash
    conda activate sctram_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/sctram; poetry lock; poetry install; conda deactivate; cd
    ```

    ```bash
    conda activate sctram_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/sctram; poetry run pre-commit run --all-files; conda deactivate; cd
    ```

    ```bash
    conda activate sctram_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/sctram/docs; rm -rf _build; poetry run make html; open _build/html/index.html; conda deactivate; cd
    ```

    ```bash
    conda activate sctram_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/sctram; poetry run pytest --typeguard-packages=sctram; conda deactivate; cd
    ```

    ```bash
    conda activate sctram_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/sctram; safety check --full-report --file=.nox/safety-3-9/tmp/requirements.txt; conda deactivate; cd
    ```