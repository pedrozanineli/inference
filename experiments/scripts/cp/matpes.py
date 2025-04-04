import os
import time
import matgl
import torch
import pickle
import warnings
import numpy as np
from ase import Atoms
from tqdm import tqdm
from ase.io import read
from ase.build import bulk
from codecarbon import EmissionsTracker
from matgl.ext.ase import PESCalculator

def forces_calculator(molecule):
    forces = []
    for atom in molecule.get_forces():
        sum = 0
        for direction in atom: sum += direction**2
        force_module = round(np.sqrt(sum),5)
        forces.append([atom,force_module])
    return forces

def inference():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    files = os.listdir('Dataset_ZrO2')
    real_energies,predicted_energies = [],[]
    real_forces,predicted_forces = [],[]

    tracker = EmissionsTracker(output_dir='emissions',output_file=f'matpes.csv' )
    tracker.start()
    begin = time.time()

    potential = matgl.load_model('TensorNet-MatPES-PBE-v2025.1-PES')
    calc = PESCalculator(potential)
    path = 'results'
    begin = time.time()

    for file in tqdm(range(len(files))):

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
    with open('run_time.txt','a') as f: f.write(f'{total_time:.6f}')
    tracker.stop()

inference()
