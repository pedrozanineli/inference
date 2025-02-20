import os
import sys
import csv
import time
import torch
import pickle
import warnings
import subprocess
import numpy as np
from ase import Atoms
from tqdm import tqdm
from ase.io import read
from ase.build import bulk
from codecarbon import EmissionsTracker

warnings.filterwarnings('ignore')

model_name = sys.argv[1]
# model_name = 'fair-chem'

def change_env(uip):
        activate_cmd = (
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {uip}"
                # f"python {script_path} {uip}"
        )
        subprocess.run(activate_cmd, shell=True, executable="/bin/bash")

def forces_calculator(molecule):
    forces = []
    for atom in molecule.get_forces():
        sum = 0
        for direction in atom: sum += direction**2
        force_module = round(np.sqrt(sum),5)
        forces.append([atom,force_module])
    return forces

def set_calc(model_name):

        # change_env(model_name)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        # ORB model
        if model_name == "orb":
                from orb_models.forcefield import pretrained
                from orb_models.forcefield.calculator import ORBCalculator

                model = pretrained.orb_v2(device=device)
                calc = ORBCalculator(model, device=device)

        # Fair-chem model
        elif model_name == "fair-chem":
                from fairchem.core import OCPCalculator
                calc = OCPCalculator(
                # model_name="2025-01-10-dpa3-openlam.pth",
                # local_cache="pretrained_models",
                checkpoint_path='pretrained_models/eqV2_86M_omat.pt',
                cpu=False,
                )

        # MACE model
        elif model_name == "mace":
                from mace.calculators import mace_mp
                # calc = mace_mp(model="large",device='cuda',default_dtype='float64')
                calc = mace_mp(model="medium-mpa-0",device='cuda',default_dtype='float64')

        # DeepMD
        elif model_name == "deepmd":
                from deepmd.calculator import DP
                model = "pretrained_models/2025-01-10-dpa3-openlam.pth"
                calc = DP(model=model)

        # Grace
        elif model_name == "grace":
                # from tensorpotential.calculator import TPCalculator
                # calc = TPCalculator('pretrained_models/GRACE-2L-OAM_28Jan25/metadata.yaml')

                from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels
                calc = grace_fm(GRACEModels.GRACE_2L_OAM_28Jan25)
                
        else:
                raise ValueError("Model not supported. The list of currently supported models is on etc/README.md")

        # ZnO2 dataset inference

        files = os.listdir('Dataset_ZrO2')
        real_energies,predicted_energies = [],[]
        real_forces,predicted_forces = [],[]

        path = f'results/{model_name}'
        if not os.path.exists(path): os.makedirs(path)

        tracker = EmissionsTracker(output_dir='emissions',output_file=f'{model_name}.csv' )
        tracker.start()
        begin = time.time()

        for file in tqdm(range(len(files))):

                begin = time.time()
                structures = read(f'Dataset_ZrO2/{files[file]}',index=':') # to read all sampled structures
                for struc_index,structure in enumerate(structures):

                        real_energy = structure.get_potential_energy()
                        real_force = forces_calculator(structure)

                        structure.calc = calc
                        predicted_energy = structure.get_potential_energy()
                        predicted_force = forces_calculator(structure)

                        # energy by atom
                        real_energies.append(real_energy/len(structure))
                        predicted_energies.append(predicted_energy/len(structure))

                        real_forces.append(real_force)
                        predicted_forces.append(predicted_force)

                with open(f'{path}/predicted_energies.pkl','wb') as f: pickle.dump(predicted_energies,f)
                with open(f'{path}/predicted_forces.pkl','wb') as f: pickle.dump(predicted_forces,f)
                with open(f'{path}/dft_forces.pkl','wb') as f: pickle.dump(real_forces,f)
                with open(f'{path}/dft_energies.pkl','wb') as f: pickle.dump(real_energies,f)

        end = time.time()
        total_time = end-begin
        with open('run_time.txt','a') as f: f.write(f'{model_name}: {total_time:.6f}')
        tracker.stop()

set_calc(model_name)
