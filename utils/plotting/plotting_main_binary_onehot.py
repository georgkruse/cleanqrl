import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from utils.config.common import extract_hyperparameters


# paths for 3x3 maze
fig_name = 'comparison_3x3_paper_binary_vs_onehot'
# title = 'comparision 3x3 Mazes'
paths = ['logs/paper/binary/3x3/qpg/2024-12-16--15-07-22_QRL_QPG+', 
         'logs/paper/binary/3x3/qdqn/2024-12-16--15-14-50_QRL_QDQN+',  
         'logs/paper/binary/3x3/qbm/2024-12-16--16-54-35_QRL_QBM',

         'logs/paper/onehot/3x3/qpg/2024-12-16--14-28-04_QRL_QPG+', 
         'logs/paper/onehot/3x3/qdqn/2024-12-16--14-42-22_QRL_QDQN+', 
         'logs/paper/onehot/3x3/qbm/2024-12-17--15-21-07_QRL_QBM'
        ]
labels = ['$QPG - binary$', '$QDQN - binary$', '$FE-QRL - binary$', '$QPG - onehot$', '$QDQN - onehot$', '$FE-QRL - onehot$']

y_lim = [-1, 1]
r = 0.7
steps = 10_000

line_styles = [':', ':', ':', '--', '--', '--']
colors = ['dodgerblue', 'darkorange', 'forestgreen', 'dodgerblue', 'darkorange', 'forestgreen']

fig, (ax_steps, ax_time) = plt.subplots(1,2, figsize=(8, 5), sharey=True)

for path, label, line_style, color in zip(paths, labels, line_styles, colors):
    print(path, label)
    folder_path = os.path.basename(os.path.normpath(path))
    results_file_name = "/result.json"
    result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
    results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

    curves = []
    curves_x_axis = []
    for i, data_exp in enumerate(results):
        if 'FE-QRL' in label:
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
    ax_steps.plot(x_axis, mean, label=label, linestyle=line_style, color=color)
    ax_steps.fill_between(x_axis, lower, upper, alpha=0.5, linestyle=line_style, color=color)


    # curves = []
    # curves_x_axis = []
    # for i, data_exp in enumerate(results):
    #     if 'FE-QRL' in label:
    #         curves.append(data_exp['episode_reward_mean'].values*0.01)
    #     else:
    #         curves.append(data_exp['episode_reward_mean'].values)
    #     curves_x_axis.append(data_exp['circuit_executions'].values)                    
    # min_length = min([len(d) for d in curves])
    # data = [d[:min_length] for d in curves]
    # x_axis = curves_x_axis[0][:min_length]
    # data = np.vstack(data)
    # mean = np.mean(data, axis=0)
    # std = np.std(data, axis=0)
    # upper = mean +  std
    # lower = mean  -  std
    # ax_ex.plot(x_axis, mean, label=label, linestyle=line_style, color=color)
    # ax_ex.fill_between(x_axis, lower, upper, alpha=0.5, linestyle=line_style, color=color)

    curves = []
    curves_x_axis = []
    for i, data_exp in enumerate(results):
        if 'FE-QRL' in label:
            curves.append(data_exp['episode_reward_mean'].values*0.01)
            curves_x_axis.append(data_exp['circuit_executions'].values*115*1e-3)                    
        elif 'QPG' in label or 'QDQN' in label:
            curves.append(data_exp['episode_reward_mean'].values)
            curves_x_axis.append(data_exp['circuit_executions'].values*(300e-9+2*300e-9*5+3*5*50e-9)*1000)   
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
    ax_time.plot(x_axis, mean, label=label, linestyle=line_style, color=color)
    ax_time.fill_between(x_axis, lower, upper, alpha=0.5, linestyle=line_style, color=color)



ax_steps.set_xlabel("$environment \\ steps$", fontsize=13)
# ax_ex.set_xlabel("circuit executions", fontsize=13)
ax_time.set_xlabel("$clock \\ time \\ [s]$", fontsize=13)


ax_steps.hlines(y=r, xmin= 0, xmax=steps, linestyles='dashdot', color='black')
ax_steps.set_xlim(0, steps)

# ax_main.axvline(x=2000, label='grover qrl', color='b', linestyle='--')

# Set the axis limits
# ax_steps.set_xlim(0, 20000)
ax_steps.set_ylim(*y_lim)
# ax_time.set_xscale('log')
# ax_steps.hlines(y=70, xmin= 0, xmax=100000, linestyles='dashdot', color='black')
# axis.set_ylim(-30, 3)
ax_steps.set_ylabel('$reward$', fontsize=13)
# ax_main.set_xlim(0, r)
ax_time.legend(fontsize=12, loc='lower right')
ax_steps.minorticks_on()
ax_steps.grid(which='both', alpha=0.4)
# ax_ex.grid(which='both', alpha=0.4)
ax_time.grid(which='both', alpha=0.4)

# plt.text(1.01, 0.6, info_text, transform=ax_main.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
# fig.suptitle(title, fontsize=15)

fig.tight_layout()
plt.savefig(f'logs/paper/{fig_name}.png') #, dpi=1200)
print('Done')