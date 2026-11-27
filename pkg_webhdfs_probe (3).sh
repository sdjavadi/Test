#!/bin/bash
#SBATCH --job-name=pkg_webhdfs_probe
#SBATCH --partition=job.gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=../log/webhdfs_probe_%j.log
# No GPU requested, so it does not wait behind the running network job (if the partition refuses a job without one,
# add:  #SBATCH --gpus=1).  Submit from ~/repos/pkg/code:  sbatch pkg_webhdfs_probe.sh
source /projects/sa15474/.bashrc
conda activate gpu-default_clone
python -u ../code/pkg_webhdfs_probe.py
