import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from utils.config.common import extract_hyperparameters



# paths for frozenlake4x4 maze
fig_name = 'comparison_frozenlake4x4_paper_no_entanglement'
title = '$Frozen \\ Lake \\ 4x4$'
paths = [
         'logs/paper/binary/frozenlake4x4/qpg/2024-12-16--10-31-36_QRL_QPG+',
         'logs/paper/binary/frozenlake4x4/qdqn/2024-12-16--11-41-55_QRL_QDQN+', 
         'logs/paper/binary/frozenlake4x4/qpg/2024-12-16--11-04-01_QRL_QPG+',
         'logs/paper/binary/frozenlake4x4/qdqn/2024-12-16--12-40-44_QRL_QDQN+', 
         'logs/paper/binary/frozenlake4x4/qpg/2024-12-16--10-50-27_QRL_QPG+',
         'logs/paper/binary/frozenlake4x4/qdqn/2024-12-16--13-10-55_QRL_QDQN+',
       ]
labels = ['$QPG$', '$QDQN$', '$QPG - no \\ ent. (A)$', '$QDQN - no \\ ent. (A)$', '$QPG - no \\ ent. (B)$', '$QDQN - no \\ ent. (B)$']
line_style = ['-', '-', ':', ':', '--', '--']
colors = ['dodgerblue', 'darkorange', 'dodgerblue', 'darkorange', 'dodgerblue', 'darkorange']
r = 1.0
steps = 20_000

fig, ax_steps = plt.subplots(1, figsize=(5, 5))

for idx, (path, label) in enumerate(zip(paths, labels)):
    print(path, label)
    folder_path = os.path.basename(os.path.normpath(path))
    results_file_name = "/result.json"
    result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
    results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

    curves = []
    curves_x_axis = []
    for i, data_exp in enumerate(results):
        if label == 'QBM':
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
    ax_steps.plot(x_axis, mean, label=label, linestyle=line_style[idx], color=colors[idx])
    ax_steps.fill_between(x_axis, lower, upper, alpha=0.5, color=colors[idx])


ax_steps.set_xlabel("$environment \\ steps$", fontsize=13)

ax_steps.hlines(y=r, xmin= 0, xmax=steps, linestyles='dashdot', color='black')
ax_steps.set_xlim(0, steps)
# ax_main.axvline(x=2000, label='grover qrl', color='b', linestyle='--')

# Set the axis limits
# ax_steps.set_xlim(0, 20000)
# ax_steps.set_ylim(-0.1, 1.3)

# ax_steps.hlines(y=70, xmin= 0, xmax=100000, linestyles='dashdot', color='black')
# axis.set_ylim(-30, 3)
ax_steps.set_ylabel('$reward$', fontsize=13)
# ax_main.set_xlim(0, r)
ax_steps.legend(fontsize=10, loc='lower right')
ax_steps.minorticks_on()
ax_steps.grid(which='both', alpha=0.4)


# plt.text(1.01, 0.6, info_text, transform=ax_main.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
fig.suptitle(title, fontsize=13)

fig.tight_layout()
plt.savefig(f'logs/paper/{fig_name}.png') #, dpi=1200)
print('Done')