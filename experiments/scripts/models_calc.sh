#!/bin/bash
#SBATCH --job-name=fm-eval
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

declare -A full_models

# full_models["deepmd"]="DPA3-v2-OpenLAM|DPA3-v2-MPtrj"
# full_models["fair-chem"]="eSEN-30M-OAM|eSEN-30M-MP|eqV2 M|eqV2 S DeNS"
# full_models["grace"]="GRACE-2L-OAM|GRACE-1L-OAM|GRACE-2L-MPtrj"
# full_models["mace"]="MACE-MPA-0|MACE-MP-0"
# full_models["mattersim"]="MatterSim-v1.0.0-5M"
# full_models["orb"]="ORB|ORB-MPTrj"
# full_models["sevenn"]="SevenNet-MF-ompa|SevenNet-l3i5"

full_models["orb"]="ORB-V3"

source ~/.bashrc

for category in "${!full_models[@]}"; do
    
    # echo "Categoria: $category"
    # conda activate $category
    conda activate orb-v3

    IFS='|' read -r -a models <<< "${full_models[$category]}"
    
    for model in "${models[@]}"; do
        
        # echo "  - Modelo: $model"

        python3 run.py $category $model

    done
done
