#!/bin/bash
#SBATCH --job-name=pkg_net_gpu
#SBATCH --partition=job.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=../log/gpu_net_%j.log
# Submit from ~/repos/pkg/code:  sbatch pkg_network_gpu.sh
# Resubmit to continue after the time limit: finished windows are skipped.
source /projects/sa15474/.bashrc
conda activate gpu-default_clone
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u ../code/pkg_network_gpu.py
