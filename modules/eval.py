import pickle
import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

colors = ['#66c5cc', '#f6cf71', '#f89c74', '#dcb0f2', '#87c55f', '#9eb9f3', '#fe88b1', '#c9db74', '#8be0a4', '#b497e7']
symbols = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'D', 'x']
code = list(itertools.zip_longest(colors, symbols, fillvalue=None))

def load(path):
    file = open(path,'rb')
    load = pickle.load(file)
    file.close()
    return load

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

def parity_plot(X,y,label,unit,style=0,model=None):

    # reg = LinearRegression().fit(X.reshape(-1, 1),y)
    # r2 = reg.score(X.reshape(-1, 1),y)
    # linreg_x = np.linspace(0.8*min(X),1.2*max(X),len(y))
    # linreg_y = reg.predict(linreg_x.reshape(-1, 1))
    # plt.plot(linreg_x,linreg_y,color='gray',lw=0.75,linestyle='--')

    line_points = np.linspace(-1e4,1e4,2)
    plt.plot(line_points,line_points,color='gray',lw=1.2,linestyle='--',alpha=0.7)
    
    rmse = metrics.root_mean_squared_error(X,y)

    style_color,style_symbol = code[style]

    if model != None:
        plt.scatter(X, y,s=18,color=style_color,marker=style_symbol,alpha=0.25,label=f'{model} (RMSE={round(rmse,2)} {unit})')
    else:
        plt.scatter(X, y,s=18,color=style_color,marker=style_symbol,alpha=0.25,label=f'RMSE={round(rmse,2)} {unit}')

    plt.ylabel(f'ML Predicted {label}')
    plt.xlabel(f'DFT {label}')
    plt.legend()

def coded_parity_plot(X,y,label,ref):

    # reg = LinearRegression().fit(X.reshape(-1, 1),y)
    # r2 = reg.score(X.reshape(-1, 1),y)
    # linreg_x = np.linspace(min(X),max(X),10)
    # linreg_y = reg.predict(linreg_x.reshape(-1, 1))
    # plt.plot(linreg_x,linreg_y,color='gray',lw=0.75) #,label=f'Rˆ2={round(r2,5)}')

    line_points = np.linspace(0.9*min(y),1.1*max(y),10)
    plt.plot(line_points,line_points,color='gray',lw=0.75) #,label=f'Rˆ2={round(r2,5)}')
    
    geometries = [i[1] for i in ref.values()]
    geometries_set = set(geometries)

    geometries_combinations = {}
    for i,geometry in enumerate(geometries_set):
        geometries_combinations[geometry] = code[i]
    
    geometry_test = []
    for i in range(len(X)):
        geometry_set = geometries_combinations[ref[i][1]]

        if ref[i][1] not in geometry_test:
            geometry_test.append(ref[i][1])
            plt.scatter(X[i],y[i],s=25,color=geometry_set[0],marker=geometry_set[1],alpha=0.75,label=ref[i][1])
        else:
            plt.scatter(X[i],y[i],s=25,color=geometry_set[0],marker=geometry_set[1],alpha=0.75)

    plt.xlabel(f'ML Predicted {label}')
    plt.ylabel(f'DFT {label}')
    plt.legend(loc='upper left')

def coded_scatter_plot(X,y,labelX,labelY,ref):
    colors = ['#66c5cc', '#f6cf71', '#f89c74', '#dcb0f2', '#87c55f', '#9eb9f3', '#fe88b1', '#c9db74', '#8be0a4', '#b497e7']
    symbols = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'D', 'x']
    code = list(itertools.zip_longest(colors, symbols, fillvalue=None))

    geometries = [i[1] for i in ref.values()]
    geometries_set = set(geometries)

    geometries_combinations = {}
    for i,geometry in enumerate(geometries_set):
        geometries_combinations[geometry] = code[i]

    geometry_test = []
    for i in range(len(X)):
        geometry_set = geometries_combinations[ref[i][1]]

        if ref[i][1] not in geometry_test:
            geometry_test.append(ref[i][1])
            plt.scatter(X[i],y[i],s=25,color=geometry_set[0],marker=geometry_set[1],alpha=0.75,label=ref[i][1])
        else:
            plt.scatter(X[i],y[i],s=25,color=geometry_set[0],marker=geometry_set[1],alpha=0.75)

    plt.xlabel(f'Predicted {labelX}')
    plt.ylabel(f'Real {labelY}')
    plt.legend()

def error_visualization(models):
    keys,errors = [],[]
    for model in models:
        
        model_key = model[0]
        dft_energies,ml_energies = model[1]
        error = []
        
        for dft_energy,ml_energy in zip(np.array(dft_energies),np.array(ml_energies)):
            # energy_absolute_error = np.abs(ml_energy - dft_energy)
            energy_absolute_error = ml_energy - dft_energy
            error.append(energy_absolute_error)
        
        keys.append(model_key)
        errors.append(error)
        
    plt.violinplot([i for i in errors],showextrema=False)
    plt.boxplot([i for i in errors],tick_labels=keys)