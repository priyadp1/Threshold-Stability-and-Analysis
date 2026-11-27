#!/bin/bash
#SBATCH --job-name=arborenum_reg
#SBATCH --output=arborenum_reg_%j.out
#SBATCH --error=arborenum_reg_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate regular (non-optimized) environment with arborenum + tree-praxis
conda activate py312

# Verify Python
python --version
python -c "import arborenum; print('arborenum OK:', arborenum.__file__)"
python -c "import praxis; print('praxis OK:', praxis.__file__)"

# Run PRAXIS, standard ArborEnum, and anytime ArborEnum experiments
bash run_praxis.sh &
pid1=$!
bash run_arborenum_standard.sh &
pid2=$!
bash run_arborenum_anytime.sh &
pid3=$!
status=0
wait $pid1 || { echo "praxis failed" >&2; status=1; }
wait $pid2 || { echo "standard failed" >&2; status=1; }
wait $pid3 || { echo "anytime failed" >&2; status=1; }
exit $status
