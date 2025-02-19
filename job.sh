#!/bin/bash
#SBATCH --job-name=dmd
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu-pbi  
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

conda init

conda activate deepmd

python3 1_calculator.py deepmd

