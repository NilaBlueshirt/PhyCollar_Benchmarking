#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores
#SBATCH -t 7-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o slurm.%A_%a.txt # file to save job's STDOUT & STDERR
#SBATCH --export=NONE   # Purge the job-submitting shell environment


# Generate the manifest file:
# parallel -k echo {} ::: $(seq 1 50) ::: cat mlp rf svm lg ::: none ::: /scratch/tianche5/PhyCollar/PC_baseline/binary/*/*/*.py > manifest_1
# parallel -k echo {} ::: $(seq 1 50) ::: cat mlp rf svm lg ::: none ::: /scratch/tianche5/PhyCollar/PC_baseline/binary/*/*.py > manifest_2
# cat manifest_1 manifest_2 > manifest
# Run the job array for sub_job 1:
# sbatch -a 1-2750 -c 16 launch_PC.sh manifest

manifest="${1:?ERROR -- must pass a manifest file}"
taskid=$SLURM_ARRAY_TASK_ID
trial_idx=$(getline $taskid $manifest | cut -f 1 -d ' ')
model=$(getline $taskid $manifest | cut -f 2 -d ' ')
pca=$(getline $taskid $manifest | cut -f 3 -d ' ')
target_location=$(getline $taskid $manifest | cut -f 4 -d ' ')

module purge
# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate tree

python PC.py $trial_idx $model $pca $target_location
