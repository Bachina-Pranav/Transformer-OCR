#!/bin/bash
#SBATCH -c 35
#SBATCH --mem-per-cpu=2048
#SBATCH --mincpus=35
#SBATCH --mail-user=sai.gunda@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --time=96:00:00
#SBATCH -A saigunda
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnode052
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ~/pageocr
git branch
wandb online
export WANDB_MODE=online
python3 pretraining.py 
