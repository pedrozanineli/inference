import pickle
import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

colors = ['#66c5cc', '#f6cf71', '#f89c74', '#dcb0f2', '#87c55f', '#9eb9f3', '#fe88b1', '#c9db74', '#8be0a4', '#b497e7']
symbols = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'D', 'x']
code = list(itertools.zip_longest(colors, symbols, fillvalue=None))

from matplotlib import font_manager

font_path = 'modules/NotoSansMath-Regular.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

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

def forces_treatment(X,y):
    X_aux,y_aux = [],[]
    for y_i in range(len(y)):
        if y[y_i] < 35:
            X_aux.append(X[y_i])
            y_aux.append(y[y_i])
    X,y = X_aux.copy(),y_aux.copy()
    return X,y

def parity_plot(X,y,label,unit,style=0,model=None):

    line_points = np.linspace(-1e3,1e3,2)
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

def latex_table_results(energies,forces,models,database,params,non_compliant):

    rows = []
    for model_energy,model_force,model,n,db,param in zip(energies,forces,models,non_compliant,database,params):
        
        e_r2,e_mae,e_rmse = eval_report(model_energy[0],model_energy[1])
        f_r2,f_mae,f_rmse = eval_report(model_force[0],model_force[1])

        row = [n,model,db,param,round(e_r2,4),round(e_mae,4),round(e_rmse,4),round(f_r2,4),round(f_mae,4),round(f_rmse,4)]
        rows.append(row)

    bold_indexes = []
    for i,row in enumerate(rows):

        if i == 0:
            e_r2,e_mae,e_rmse,f_r2,f_mae,f_rmse = row[4:]
            bold_indexes = 6*[0]
        else:
            e_r2_aux,e_mae_aux,e_rmse_aux,f_r2_aux,f_mae_aux,f_rmse_aux = row[4:]

        def compare(curr,aux,curr_index,index,minor=True):
            if minor:
                if curr > aux:
                    curr = aux
                    bold_indexes[curr_index] = index
            else:
                if curr < aux:
                    curr = aux
                    bold_indexes[curr_index] = index
            return curr
        
        if i != 0:
            e_r2 = compare(e_r2,e_r2_aux,0,i,minor=False)
            e_mae = compare(e_mae,e_mae_aux,1,i,minor=True)
            e_rmse = compare(e_rmse,e_rmse_aux,2,i,minor=True)
            f_r2 = compare(f_r2,f_r2_aux,3,i,minor=False)
            f_mae = compare(f_mae,f_mae_aux,4,i,minor=True)
            f_rmse = compare(f_rmse,f_rmse_aux,5,i,minor=True)
            
    for i,row in enumerate(rows):
        
        n,model,db,param,e_r2,e_mae,e_rmse,f_r2,f_mae,f_rmse = row
        
        for j,value in enumerate(row):

            if j == 0: 
                if n: print(f'{model} $\star$',end=' & ')
                else: print(f'{model}',end=' & ')
            
            if j > 1:
                if i == bold_indexes[j-4]:
                    if j+1 != len(row): print('\\textbf','{',value,'}',end=' & ')
                    else: print('\\textbf','{',value,'}',end='\\\\')
                elif j+1 != len(row): print(value,end=' & ')
                else: print(value,end='\\\\')

            # elif j != 0:
                # if j-1 == len(row): print(value,end=' & ')
                # else: print(value)

        print()