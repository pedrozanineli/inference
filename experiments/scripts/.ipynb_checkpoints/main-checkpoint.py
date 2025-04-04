import os
import sys
import subprocess

sys.path.append(os.path.abspath('..'))

full_models = {
    # "fair-chem": ['eSEN-30M-OAM', 'eSEN-30M-MP', 'eqV2 M', 'eqV2 S DeNS'],
    
    
    # "deepmd": ['DPA3-v2-OpenLAM', 'DPA3-v2-MPtrj'],
    # "grace": ['GRACE-2L-OAM', 'GRACE-1L-OAM', 'GRACE-2L-MPtrj'],
    
    "grace": ['GRACE-1L-OAM', 'GRACE-2L-MPtrj'],
    "mace": ['MACE-MPA-0', 'MACE-MP-0'],
    "mattersim": ['MatterSim-v1.0.0-5M'],
    "orb": ['ORB', 'ORB MPTrj'],
    "sevenn": ['SevenNet-MF-ompa', 'SevenNet-l3i5'],
}

for calculator in full_models:
    for model in full_models[calculator]:
        subprocess.run(f'python3 environment_change.py {calculator} {model}',shell=True)

