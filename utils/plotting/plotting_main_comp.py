import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from utils.config.common import extract_hyperparameters


labels = ['$QPG$', '$QDQN$', '$FE-QRL$', '$AA-QRL$']
colors = ['dodgerblue', 'darkorange', 'forestgreen', 'indianred']

# paths for 3x3 maze
# fig_name = 'comparison_3x3_paper'
# title = '$Maze \\ 3x3$'
# paths = ['logs/paper/binary/3x3/qpg/2024-12-16--15-07-22_QRL_QPG+', 
#          'logs/paper/binary/3x3/qdqn/2024-12-16--15-14-50_QRL_QDQN+', 
#          'logs/paper/onehot/3x3/qbm/2024-12-17--15-21-07_QRL_QBM',
#          'logs/paper/3x3/grover/2024-12-16--18-02-57_QRL_Grover']
# y_lim = [-1, 1]
# r = 0.7 
# steps = 10_000

# paths for 3x5 maze
# fig_name = 'comparison_3x5_paper'
# title = '$Maze \\ 3x5$'
# paths = ['logs/paper/binary/3x5/qpg/2024-12-16--15-51-20_QRL_QPG+', 
#          'logs/paper/binary/3x5/qdqn/2024-12-16--16-29-07_QRL_QDQN+', 
#          'logs/paper/onehot/3x5/qbm/2024-12-17--15-02-52_QRL_QBM',
#          'logs/paper/3x5/grover/2024-12-16--18-09-58_QRL_Grover']
# y_lim = [-1, 0.7]
# r = 0.5 
# steps = 20_000

# paths for frozenlake4x4 maze
# fig_name = 'comparison_frozenlake4x4_paper'
# title = '$Frozen \\ Lake \\ 4x4$'
# paths = ['logs/paper/binary/frozenlake4x4/qpg/2024-12-16--10-31-36_QRL_QPG+',
#          'logs/paper/binary/frozenlake4x4/qdqn/2024-12-16--11-41-55_QRL_QDQN+', 
#          'logs/paper/onehot/frozenlake4x4/qbm/2024-12-17--07-19-55_QRL_QBM',
#          'logs/paper/frozenlake4x4/grover/2024-12-16--18-19-39_QRL_Grover']
# y_lim = [0.0, 1.05]
# r = 1.0 
# steps = 20_000

# paths for frozenlake8x8 maze
fig_name = 'comparison_frozenlake8x8_paper'
title = '$Frozen \\ Lake \\ 8x8$'
paths = ['logs/paper/binary/frozenlake8x8/qpg/2024-12-16--18-23-47_QRL_QPG+',
         'logs/paper/binary/frozenlake8x8/qdqn/2024-12-13--23-30-10_QRL_QDQN+', 
         'logs/paper/frozenlake8x8/qbm/2024-12-05--12-30-57_QRL_QBM',
         'logs/paper/frozenlake8x8/grover/2024-12-16--18-24-59_QRL_Grover']
y_lim = [0.0, 1.05]
r = 1.0 
steps = 100_000

fig, (ax_steps, ax_ex, ax_time) = plt.subplots(1,3, figsize=(9, 4), sharey=True)

for path, label, color in zip(paths, labels, colors):
    print(path, label)
    folder_path = os.path.basename(os.path.normpath(path))
    results_file_name = "/result.json"
    result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
    results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

    curves = []
    curves_x_axis = []
    for i, data_exp in enumerate(results):
        if label == '$FE-QRL$':
            curves.append(data_exp['episode_reward_mean'].values*0.01)
        else:
            curves.append(data_exp['episode_reward_mean'].values)
        curves_x_axis.append(data_exp['num_env_steps_sampled'].values)                    
    min_length = min([len(d) for d in curves])
    data = [d[:min_length] for d in curves]
    x_axis = curves_x_axis[0][:min_length]
    data = np.vstack(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    upper = mean +  std
    lower = mean  -  std
    ax_steps.plot(x_axis, mean, label=label, color=color)
    ax_steps.fill_between(x_axis, lower, upper, alpha=0.5, color=color)


    curves = []
    curves_x_axis = []
    for i, data_exp in enumerate(results):
        if label == '$FE-QRL$':
            curves.append(data_exp['episode_reward_mean'].values*0.01)
        else:
            curves.append(data_exp['episode_reward_mean'].values)
        curves_x_axis.append(data_exp['circuit_executions'].values)                    
    min_length = min([len(d) for d in curves])
    data = [d[:min_length] for d in curves]
    x_axis = curves_x_axis[0][:min_length]
    data = np.vstack(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    upper = mean +  std
    lower = mean  -  std
    ax_ex.plot(x_axis, mean, label=label, color=color)
    ax_ex.fill_between(x_axis, lower, upper, alpha=0.5, color=color)

    curves = []
    curves_x_axis = []
    for i, data_exp in enumerate(results):
        if label == '$FE-QRL$':
            curves.append(data_exp['episode_reward_mean'].values*0.01)
            curves_x_axis.append(data_exp['circuit_executions'].values*115*1e-3)                    
        elif label in ['$QPG$', '$QDQN$']:
            curves.append(data_exp['episode_reward_mean'].values)
            curves_x_axis.append(data_exp['circuit_executions'].values*(300e-9+2*300e-9*5+3*5*50e-9)*1000)   
        else:
            curves.append(data_exp['episode_reward_mean'].values)
            curves_x_axis.append(data_exp['circuit_executions'].values*(300e-9+2*300e-9+4*50e-9)*1000)              
    min_length = min([len(d) for d in curves])
    data = [d[:min_length] for d in curves]
    x_axis = curves_x_axis[0][:min_length]
    data = np.vstack(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    upper = mean +  std
    lower = mean  -  std
    ax_time.plot(x_axis, mean, label=label, color=color)
    ax_time.fill_between(x_axis, lower, upper, alpha=0.5, color=color)

ax_steps.set_xlabel("$environment \\ steps$", fontsize=13)
ax_ex.set_xlabel("$circuit \\ executions$", fontsize=13)
ax_time.set_xlabel("$clock \\ time \\ [s]$", fontsize=13)



# ax_steps.axvline(x=2000, label='grover qrl', color='b', linestyle='--')

# Set the axis limits
ax_steps.set_ylim(*y_lim)

ax_steps.hlines(y=r, xmin= 0, xmax=steps, linestyles='dashdot', color='black')
ax_steps.set_xlim(-100, steps)
# ax_ex.hlines(y=r, xmin= 0, xmax=steps, linestyles='dashdot', color='black')
# ax_ex.set_xlim(0, steps)
# ax_time.hlines(y=r, xmin= 0, xmax=steps, linestyles='dashdot', color='black')
# ax_time.set_xlim(0, steps)
ax_steps.set_ylabel('$reward$', fontsize=13)
ax_steps.legend(fontsize=12, loc='lower right')
ax_steps.minorticks_on()
ax_steps.grid(which='both', alpha=0.4)
ax_ex.grid(which='both', alpha=0.4)
ax_time.grid(which='both', alpha=0.4)
ax_ex.set_xscale('log')
ax_time.set_xscale('log')

# plt.text(1.01, 0.6, info_text, transform=ax_main.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
fig.suptitle(title, fontsize=15)

fig.tight_layout()
plt.savefig(f'logs/paper/{fig_name}.png') #, dpi=1200)
print('Done')