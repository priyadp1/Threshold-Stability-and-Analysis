#!/bin/bash
#SBATCH --job-name=arborenum
#SBATCH --output=arborenum_%j.out
#SBATCH --error=arborenum_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate Python 3.12 environment
conda activate py312

# Verify Python
python --version

# Run ArborEnum experiments
bash run_experiments.sh