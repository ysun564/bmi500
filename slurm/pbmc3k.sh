#!/bin/bash -l
#SBATCH -J pbmc3k
#SBATCH -p overflow
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=outputs/pbmc3k_slurm_%j.out
#SBATCH --error=outputs/pbmc3k_slurm_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load bmi/python-3.12.12
source "$HOME/venvs/bmi500-py312/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

mkdir -p outputs results

/usr/bin/time -v \
  -o outputs/pbmc3k_coarse_time.txt \
  python scanpy_pbmc.py \
    --data-dir data \
    --data-set pbmc3k \
    --out-dir results \
    --num-threads 1 \
  > outputs/pbmc3k_coarse_console.txt 2>&1

python scanpy_pbmc_instrumented.py \
  --data-dir data \
  --data-set pbmc3k \
  --out-dir results \
  --num-threads 1 \
  > outputs/pbmc3k_instrumented_console.txt 2>&1

python scanpy_pbmc_cprofile.py \
  --data-dir data \
  --data-set pbmc3k \
  --out-dir results \
  --num-threads 1 \
  > outputs/pbmc3k_cprofile_console.txt 2>&1
