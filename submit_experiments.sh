#!/bin/bash
#SBATCH --job-name=arborenum
#SBATCH --output=arborenum_%j.out
#SBATCH --error=arborenum_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate ArborEnum (optimized) environment
conda activate arbor_opt

# Verify Python
python --version

# Build/install the vendored ArborEnum (with the optimized C++ core) if missing
if ! python -c "import arborenum.optimized" 2>/dev/null; then
    echo "Installing vendored ArborEnum from ./ArborEnum ..."
    python -m pip install ./ArborEnum
fi
python -c "import arborenum.optimized; print('arborenum OK:', arborenum.__file__)"

# Run ArborEnum experiments
bash run_arborenum_anytime_opt.sh &
pid1=$!
bash run_arborenum_standard_opt.sh &
pid2=$!
status=0
wait $pid1 || { echo "anytime failed" >&2; status=1; }
wait $pid2 || { echo "standard failed" >&2; status=1; }
exit $status