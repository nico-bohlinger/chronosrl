#!/bin/bash

#SBATCH --job-name=chronosrl
#SBATCH --output=%x_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=1000-1003

python experiment.py \
    --algorithm.name=chronosrl \
    --algorithm.depth=8 \
    --environment.name=ant_u4_maze \
    --environment.seed=$SLURM_ARRAY_TASK_ID \
    --runner.exp_name=ant_u4_maze_depth_8
