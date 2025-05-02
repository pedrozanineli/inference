import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

calculator,model = sys.argv[1],sys.argv[2]

from modules.calculator import Calculator
from modules.inference import inference

calc = Calculator.get_calculator(calculator,model)

path = '/home/p.zanineli/work/inference/experiments/zirconium-dioxide/dataset_zro2'
save_path = '/home/p.zanineli/work/inference/experiments/zirconium-dioxide/'

# inference(path,save_path,calc,calculator,model=model,track=False)
