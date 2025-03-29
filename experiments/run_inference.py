import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

calculator,model = sys.argv[1],sys.argv[2]

from modules.calculator import Calculator
from modules.inference import inference

calculator = Calculator.get_calculator(calculator,model)
print(calculator)

# def inference(path,calculator,model=None,track=False):
