#!/usr/bin/env bash

#SBATCH -J mlip
#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=180gb
#SBATCH --account dmobley_lab_gpu
#SBATCH --gres=gpu:1
#SBATCH --export ALL
#SBATCH --requeue
# Set the output and error output paths.
#SBATCH -o  slurm-%J.out
#SBATCH -e  slurm-%J.err
#

source ~/.bashrc
start=`date +%s`
conda activate /dfs9/dmobley-lab/pbehara/conda-env/mlip_optimizer

python optimize_single_model.py ./inputs/CONFIG_JSON

end=`date +%s`
runtime=$((end-start))
printf -v formatted_runtime "%02d:%02d:%02d" $((runtime/3600)) $(( (runtime%3600)/60 )) $((runtime%60))
echo "Time taken for sorting: $formatted_runtime"
echo "done"

