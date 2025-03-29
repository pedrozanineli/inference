import os;
if os.getcwd()[-9:] != 'inference': os.chdir('..')

from modules.eval import *
from modules.inference import inference
from modules.calculator import Calculator

full_models = {
    "deepmd": ['DPA3-v2-OpenLAM', 'DPA3-v2-MPtrj'],
    "fair-chem": ['eSEN-30M-OAM', 'eSEN-30M-MP', 'eqV2 M', 'eqV2 S DeNS'],
    "grace": ['GRACE-2L-OAM (default)', 'GRACE-1L-OAM', 'GRACE-2L-MPtrj'],
    "mace": ['MACE-MPA-0', 'MACE-MP-0'],
    "mattersim": ['MatterSim-v1.0.0-5M'],
    "orb": ['ORB', 'ORB MPTrj'],
    "sevenn": ['SevenNet-MF-ompa', 'SevenNet-l3i5'],
}

for calculator in full_models:

    for model in full_models[calculator]:

        activate_cmd = (
        	f"source $(conda info --base)/etc/profile.d/conda.sh && "
        	f"conda activate {calculator} && "
        	f"conda info --env"
        )

        subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
        calc = Calculator.get_calculator(calculator, model_name=model)