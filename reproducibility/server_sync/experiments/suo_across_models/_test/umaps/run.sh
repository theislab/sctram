#!/bin/bash

# Path to the Python files
DIRECTORY="/home/icb/kemal.inecik/work/codes/sctram/reproducibility/server_sync/umaps"

# Activate the Python environment
source activate sctram_dev_env

# Loop over all Python files in the directory
for file in $DIRECTORY/*.py
do
    # Extract the basename without the extension
    BASENAME=$(basename "$file" .py)

    # Create a SLURM script for each Python file with a specific name
    SCRIPT_NAME="$DIRECTORY/${BASENAME}_slurm_job.sh"
    cat << EOF > $SCRIPT_NAME
#!/bin/bash

#SBATCH -J sctram_umap
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH -c 32
#SBATCH --mem=90G
#SBATCH --nice=0
#SBATCH -t 9:59:00

source activate sctram_dev_env
python $file

EOF

    # Submit the job
    sbatch $SCRIPT_NAME

sleep 1
done
