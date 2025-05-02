import pickle
import os, sys
import subprocess
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

os.chdir('..')

from modules.eval import load,forces_load,eval_report

def load_module(path,prop):
    dft,ml = load(f'{path}/dft_{prop}.pkl'),load(f'{path}/predicted_{prop}.pkl')
    return [ml,dft]

def prop_dict(structures,prop):

    values_dict,metrics_dict = {},{}

    for struc,dft,ml in zip(structures,prop[0],prop[1]):
        if struc[0] not in values_dict:
            values_dict[struc[0]] = [[dft],[ml]]
        else:
            values_dict[struc[0]][0].append(dft)
            values_dict[struc[0]][1].append(ml)
    
    for key,value in values_dict.items():

        r2,mae,rmse = eval_report(value[0],value[1])
        metrics_dict[key] = [r2,mae,rmse]
    
    return metrics_dict

path = '/home/p.zanineli/work/inference/experiments/zirconium-dioxide/results'
mlips = os.listdir(path)

if './ipycheckpoints' in mlips: mlips.remove('./ipycheckpoints')
if 'metrics.csv' in mlips: mlips.remove('metrics.csv')
if 'emissions' in mlips: mlips.remove('emissions')

model_count = []
dict_data = {}
predictions_data = {}
e_metrics_geometries,f_metrics_geometries = {},{}

for mlip in mlips:
    models = os.listdir(f'{path}/{mlip}')
    
    for model in models:
        
        model_path = f'{path}/{mlip}/{model}'
        
        structures = load(f'{model_path}/structures.pkl')
        energies = load_module(model_path,'energies')
        forces = load_module(model_path,'forces')

        force_modules = []
        for force in forces:
            force_module,x,y,z = forces_load(force)
            force_modules.append(force_module)
        
        e_r2,e_mae,e_rmse = eval_report(energies[0],energies[1])
        f_r2,f_mae,f_rmse = eval_report(force_modules[0],force_modules[1])

        e_geometries = prop_dict(structures,energies)
        f_geometries = prop_dict(structures,force_modules)

        for key in e_geometries.keys():
            e_metrics_geometries[model] = e_geometries
            f_metrics_geometries[model] = f_geometries
        
        dict_data[model] = [e_r2,e_mae,e_rmse,f_r2,f_mae,f_rmse]
        predictions_data[model] = {
            'pred_energies': energies[0],
            'dft_energies': energies[1],
            'pred_forces': force_modules[0],
            'dft_forces': force_modules[1],
        }

        model_count.append(mlip)

#

def set_final_metrics(metrics_geometries):

    final_metrics = {}

    keys = list(metrics_geometries.keys())
    geometries = list(metrics_geometries[keys[0]].keys())

    for model in metrics_geometries:
        
        final_metrics[model] = []
        model_dict = metrics_geometries[model]
            
        for geometry in geometries:

            rmse = model_dict[geometry][-1]
            final_metrics[model].append(rmse)
    
    return final_metrics

e_results = set_final_metrics(e_metrics_geometries)
f_results = set_final_metrics(f_metrics_geometries)

keys = list(e_metrics_geometries.keys())
geometries = list(e_metrics_geometries[keys[0]].keys())

df_energies = pd.DataFrame(e_results).T
df_energies.columns = geometries

df_forces = pd.DataFrame(f_results).T
df_forces.columns = geometries

df_energies.to_csv(f'{path}/geometries_energies.csv',index=False)
df_forces.to_csv(f'{path}/geometries_forces.csv',index=False)

#

df = pd.DataFrame(dict_data).T
df.columns = ['e_r2','e_mae','e_rmse','f_r2','f_mae','f_rmse']

df = df.reset_index().rename(columns={'index': 'Name'})

df.insert(1, 'Label', model_count)

multi_columns = [
    ('Model', 'Name'), ('Model', 'Label'),
    ('Energy', 'e_r2'), ('Energy', 'e_mae'), ('Energy', 'e_rmse'),
    ('Forces', 'f_r2'), ('Forces', 'f_mae'), ('Forces', 'f_rmse')
]

df.columns = pd.MultiIndex.from_tuples(multi_columns)

df.to_csv(f'{path}/metrics.csv',index=False)

st.set_page_config(layout='wide')

st.title('zirconium-dioxide')

st.subheader('metrics')

numeric_columns = [
    ('Energy', 'e_r2'), ('Energy', 'e_mae'), ('Energy', 'e_rmse'),
    ('Forces', 'f_r2'), ('Forces', 'f_mae'), ('Forces', 'f_rmse')
]

format_dict = {col: '{:.4f}' for col in numeric_columns}

styled_df = (
    df.style
    .format(format_dict)
    .background_gradient(subset=numeric_columns, cmap='Blues')
)

st.dataframe(styled_df, use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    st.subheader('energies')

    fig_energy = go.Figure()

    for model, values in predictions_data.items():
        fig_energy.add_trace(go.Scatter(
            x=values['pred_energies'],
            y=values['dft_energies'],
            mode='markers',
            name=model,
            hovertemplate=f'{model}<br>true: %{{x}}<br>predicted: %{{y}}<extra></extra>'
        ))

    full_energies = sum([values['dft_energies'] for valores in predictions_data.values()], [])
    fig_energy.add_trace(go.Scatter(
        x=full_energies,
        y=full_energies,
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='y = x'
    ))

    fig_energy.update_layout(
        xaxis_title='predicted energy',
        yaxis_title='true energy',
        height=500,
        legend_title='model'
    )

    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    st.subheader('forces')

    fig_forces = go.Figure()

    for model, values in predictions_data.items():
        fig_forces.add_trace(go.Scatter(
            x=values['pred_forces'],
            y=values['dft_forces'],
            mode='markers',
            name=model,
            hovertemplate=f'{model}<br>true: %{{x}}<br>predicted: %{{y}}<extra></extra>'
        ))

    full_forces = sum([values['dft_forces'] for values in predictions_data.values()], [])
    fig_forces.add_trace(go.Scatter(
        x=full_forces,
        y=full_forces,
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='y = x'
    ))

    fig_forces.update_layout(
        xaxis_title='predicted force',
        yaxis_title='true force',
        height=500,
        legend_title='model'
    )

    st.plotly_chart(fig_forces, use_container_width=True)
