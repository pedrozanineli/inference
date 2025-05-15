import os
import sys

calculator,model = sys.argv[1],sys.argv[2]

print(calculator,model)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.inference import inference
from modules.calculator import Calculator

calc = Calculator.get_calculator(calculator,model)

# cluster
# path = '/home/p.zanineli/work/inference/experiments/zirconium-dioxide/dataset_zro2'
# save_path = '/home/p.zanineli/work/inference/experiments/zirconium-dioxide'

# workstation
path = '/mnt/md0/home/pedro.zanineli/work/inference/experiments/zirconium-dioxide/dataset_zro2'
save_path = '/mnt/md0/home/pedro.zanineli/work/inference/experiments/zirconium-dioxide'

inference(path,save_path,calc,calculator,model=model,track=False)

# from ase.build import bulk
# a3 = bulk('Cu', 'fcc', a=3.6, cubic=True)
# a3.calc = calc
# print(a3.get_potential_energy())

