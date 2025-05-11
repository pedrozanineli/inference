# Machine Learning Interatomic Potentials Evaluation

`inference` repository is a Machine Learning Interatomic Potentials (MLIPs) evaluation for a specific dataset. If one desires to evaluate the models, it is mandatory the existence of Conda environments with the specific package installed.

On the modules folder, the three main Python files can be found: 

1. `calculator.py` - this file contains the class `Calculator` with the main function `get_calculator`. When calling it, the `calculator_name` must be passed as an argument and the `model_name` if necessairy. If the model is not defined, a default model is choosen;
2. `inference.py` - in the present file, the function `inference` receives the dataset and saving path, the calculator (i.e Mace) and the model name (i.e Mace-MP-0) as an optional argument. Additionally, the variable `track` can be used for using `codecarbon` to track the hardware usage during the inference;
3. `eval.py` - once the inference is done, the evaluation Python file contains the functions responsible for plotting the results. Example of the possible plots are the parity plot, error visualization, and parity plot coded by geometry.

In the `experiments > scripts` folder, the `models_calc.sh` contains the default file for generating the calculators and running the inference with the available calculators and models.

---

### Usage of the repository

A calculator can be defined using the class `Calculator` using the following structure, considering the variable `calculator` is the package name defined in the Conda environment and `model` the pre-trained model:

```python
calc = Calculator.get_calculator(calculator,model)
```

The returned calculator can be subsequently inputed into the `inference` function using:

```python
inference(path,save_path,calc,calculator,model=model,track=True)
```

---

### Checking the available models

For checking the available models, it is possible to use the XXX
