#!/bin/bash
#SBATCH --job-name=arborenum
#SBATCH --output=arborenum_%j.out
#SBATCH --error=arborenum_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate Python 3.12 environment
conda activate py312

# Verify Python
python --version

# Run ArborEnum experiments
bash run_arborenum_anytime_opt.sh &
pid1=$!
bash run_arborenum_standard_opt.sh &
pid2=$!
status=0
wait $pid1 || { echo "anytime failed" >&2; status=1; }
wait $pid2 || { echo "standard failed" >&2; status=1; }
exit $status