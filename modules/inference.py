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

warnings.filterwarnings('ignore')

default_calculator = {
    "deepmd": 'DPA3-v2-OpenLAM',
    "fair-chem": 'eqV2 M',
    "grace": 'GRACE-2L-OAM',
    "mace": 'MACE-MPA-0',
    "mattersim": 'MatterSim-v1.0.0-5M',
    "orb": 'ORB',
    "sevenn": 'SevenNet-MF-ompa',
}

def inference(path,saving_path,calc,calculator,model=None,track=False):

    def forces_calculator(molecule):
        forces = []
        for atom in molecule.get_forces():
            sum = 0
            for direction in atom: sum += direction**2
            force_module = round(np.sqrt(sum),5)
            forces.append([atom,force_module])
        return forces
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"using device {device}...")

    files = os.listdir(path)
    if '.ipynb_checkpoints' in files: files.remove('.ipynb_checkpoints')
    
    if not model: model = default_calculator[calculator]
    
    # mudar aqui
    results_path = f'{saving_path}/results/{calculator}/{model}'
    if not os.path.exists(results_path): os.makedirs(results_path)

    real_energies,predicted_energies = [],[]
    real_forces,predicted_forces = [],[]
    
    begin = time.time()

    if track:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(output_dir='emissions',output_file=f'{calculator}-{model}.csv',allow_multiple_runs=True)
        tracker.start()
    
    for file in tqdm(range(len(files))):

        structures = read(f'{path}/{files[file]}',index=':') # to read all sampled structures
        
        for struc_index,structure in enumerate(structures):

            real_energy = structure.get_potential_energy()
            real_force = forces_calculator(structure)

            structure.calc = calc
            predicted_energy = structure.get_potential_energy()
            predicted_force = forces_calculator(structure)

            real_energies.append(real_energy/len(structure))
            predicted_energies.append(predicted_energy/len(structure))

            real_forces.append(real_force)
            predicted_forces.append(predicted_force)

            with open(f'{results_path}/predicted_energies.pkl','wb') as f: pickle.dump(predicted_energies,f)
            with open(f'{results_path}/predicted_forces.pkl','wb') as f: pickle.dump(predicted_forces,f)
            with open(f'{results_path}/dft_forces.pkl','wb') as f: pickle.dump(real_forces,f)
            with open(f'{results_path}/dft_energies.pkl','wb') as f: pickle.dump(real_energies,f)

    end = time.time()
    total_time = end-begin
    with open('run_time.txt','a') as f: f.write(f'{calculator}-{model}: {total_time:.6f}\n')
    
    if track: tracker.stop()
