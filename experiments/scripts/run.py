import os
import sys

calculator,model = sys.argv[1],sys.argv[2]

print(calculator,model)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.inference import inference
from modules.calculator import Calculator

calc = Calculator.get_calculator(calculator,model)

path = '/home/p.zanineli/work/inference/experiments/zirconium-dioxide/dataset_zro2'
save_path = '/home/p.zanineli/work/inference/experiments/zirconium-dioxide'

inference(path,save_path,calc,calculator,model=model,track=True)
