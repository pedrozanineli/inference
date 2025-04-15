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
from sklearn import metrics

warnings.filterwarnings('ignore')

model_name = sys.argv[1]

def forces_calculator(molecule):
    forces = []
    for atom in molecule.get_forces():
        sum = 0
        for direction in atom: sum += direction**2
        force_module = round(np.sqrt(sum),5)
        forces.append([atom,force_module])
    return forces

def eval_report(y_pred,y_real):
    r2 = round(metrics.r2_score(y_real,y_pred),10)
    mae = metrics.mean_absolute_error(y_real,y_pred)
    rmse = metrics.root_mean_squared_error(y_pred,y_real)

    return r2,mae,rmse

def forces_load(forces):
    force_module,x,y,z=[],[],[],[]
    for force in forces:
        force_module.append(force[0][1])
        forces_atom = force[0][0]

        x.append(forces_atom[0])
        y.append(forces_atom[1])
        z.append(forces_atom[2])
    return force_module,x,y,z

def set_calc(model_name):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        from mace.calculators import mace_mp
        calc = mace_mp(model=model_name,device='cuda',default_dtype='float64')

        files = os.listdir('../zirconium-dioxide/dataset_zro2')
        if '.ipynb_checkpoints' in files: files.remove('.ipynb_checkpoints')

        real_energies,predicted_energies = [],[]
        real_forces,predicted_forces = [],[]

        # path = f'results/{model_name}'
        # if not os.path.exists(path): os.makedirs(path)

        begin = time.time()

        for file in tqdm(range(len(files))):

                begin = time.time()
                structures = read(f'../zirconium-dioxide/dataset_zro2/{files[file]}',index=':')
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

        e_r2,e_mae,e_rmse = eval_report(predicted_energies,real_energies)

        p_force_module,_,_,_ = forces_load(predicted_forces)
        t_force_module,_,_,_ = forces_load(real_forces)
       
        f_r2,f_mae,f_rmse = eval_report(p_force_module,t_force_module)
        
        print('energy (ev/atom)')
        print(e_r2,e_mae,e_rmse)

        print()

        print('forces (ev/A)')
        print(f_r2,f_mae,f_rmse)

        print()

        end = time.time()
        total_time = end-begin
        print(total_time)

set_calc(model_name)
