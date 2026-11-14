#!/bin/bash
#SBATCH --job-name=bw_toga2
#SBATCH --partition=datamonkey
#SBATCH --exclude=node15
#SBATCH --array=0-39%40
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=${BUSHWHACKER_WORK_DIR:-./work/toga2_nt}/logs/pool_%a.out
#SBATCH --error=${BUSHWHACKER_WORK_DIR:-./work/toga2_nt}/logs/pool_%a.err

export RAYON_NUM_THREADS=2
python3 ${BUSHWHACKER_WORK_DIR:-./work/toga2_nt}/scripts/benchmarks/run_toga2_bushwhacker_all.py $SLURM_ARRAY_TASK_ID 16 640
