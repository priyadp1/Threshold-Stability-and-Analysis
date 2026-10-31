#!/bin/bash
#SBATCH --job-name=algorithm_1
#SBATCH --output=algorithm_1_%j.out
#SBATCH --error=algorithm_1_%j.err
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

# Run Algorithm 1 on the ArborEnum results
bash run_algorithm_1.sh
