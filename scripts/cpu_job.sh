#!/bin/bash
#SBATCH --job-name=fairchem
#SBATCH --output=fc_output.log
#SBATCH --error=fc_error.log
#SBATCH --partition=cpu-pbi  
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

conda init

conda activate fair-chem

python3 1_calculator.py fair-chem

