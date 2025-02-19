#!/bin/bash
#SBATCH --job-name=fc
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu-pbi  
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

conda init

conda activate fair-chem

# python3 1_calculator.py fair-chem 

CUDA_LAUNCH_BLOCKING=1 python3 /home/pedro.zanineli/phd/inference/1_calculator.py

