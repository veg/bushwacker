#!/bin/bash
#SBATCH --job-name=bw_sim_bench
#SBATCH --partition=all
#SBATCH --array=0-179
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=${EXTERNAL_CORPORA_DIR:-./external_corpora}/logs/sim_%A_%a.out
#SBATCH --error=${EXTERNAL_CORPORA_DIR:-./external_corpora}/logs/sim_%A_%a.err

set -euo pipefail

cd ${EXTERNAL_CORPORA_DIR:-./external_corpora}
export RAYON_NUM_THREADS=4
python3 scripts/run_silverback_sim_worker.py "${SLURM_ARRAY_TASK_ID}"
