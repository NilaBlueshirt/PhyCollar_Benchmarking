#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores
#SBATCH -t 7-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o slurm.%A_%a.txt # file to save job's STDOUT & STDERR
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tianche5@asu.edu


# e.g. parallel -k echo {} ::: $(seq 1 10) ::: cat mlp rf svm > manifest_auc_col
# e.g. sbatch -a 1-40 -c 16 launch_auc_col_array.sh manifest_auc_col

manifest="${1:?ERROR -- must pass a manifest file}"
taskid=$SLURM_ARRAY_TASK_ID
trial_idx=$(getline $taskid $manifest | cut -f 1 -d ' ')
model=$(getline $taskid $manifest | cut -f 2 -d ' ')

module purge
# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate tree

python auc_col.py $trial_idx $model