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

model_name = sys.argv[1]

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
		model_name="EquiformerV2-31M-S2EF-OC20-All+MD",  # "eqV2_dens_153M_mp",
		local_cache="pretrained_models",
		cpu=False,
		# cpu = (device=='cpu')
		)

	# MACE model
	elif model_name == "mace":
		from mace.calculators import mace_mp
		calc = mace_mp(model="large",device='cuda',default_dtype='float64')

	elif model_name == "deepmd":
		from deepmd.calculator import DP
		model = "pretrained_models/2025-01-10-dpa3-openlam.pth"
		calc = DP(model=model)

	else:
		raise ValueError("Model not supported. The list of currently supported models is on etc/README.md")

	# ZnO2 dataset inference

	files = os.listdir('Dataset_ZrO2')
	real_energies,predicted_energies = [],[]
	real_forces,predicted_forces = [],[]

	for file in tqdm(range(len(files))):

		begin = time.time()
		structures = read(f'Dataset_ZrO2/{files[file]}',index=':') # to read all sampled structures
		print(files[file],len(structures))

		for struc_index,structure in enumerate(structures):

			# real_energy = structure.get_potential_energy()
			# real_force = forces_calculator(structure)

			structure.calc = calc
			predicted_energy = structure.get_potential_energy()
			predicted_force = forces_calculator(structure)

			# print(struc_index,real_energy,predicted_energy)

			# real_energies.append(real_energy)
			predicted_energies.append(predicted_energy)

			# real_forces.append(real_force)
			predicted_forces.append(predicted_force)

			# for atom in structure:

				# real_energy = atom.get_potential_energy()
				# atom.calc = calc
				# predicted_energy = atom.get_potential_energy()

				# real_energies.append(real_energy)
				# predicted_energies.append(predicted_energy)

		with open(f'results/new_predicted_energies_{model_name}.pkl','wb') as f: pickle.dump(predicted_energies,f)
		with open(f'results/new_predicted_forces_{model_name}.pkl','wb') as f: pickle.dump(predicted_forces,f)

		end = time.time()
		print(end-begin)

	# with open('results/predicted_forces_{model_name}.pkl','wb') as f: pickle.dump(predicted_forces,f)

set_calc(model_name)
