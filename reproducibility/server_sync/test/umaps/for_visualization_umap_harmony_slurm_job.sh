#!/bin/bash

#SBATCH -J sctram_umap
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 32
#SBATCH --mem=90G
#SBATCH --nice=0
#SBATCH -t 9:59:00

source activate sctram_dev_env
python /home/icb/kemal.inecik/work/codes/sctram/reproducibility/server_sync/umaps/for_visualization_umap_harmony.py

