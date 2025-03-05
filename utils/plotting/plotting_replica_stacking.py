import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from utils.config.common import extract_hyperparameters



# paths for frozenlake4x4 maze
fig_name = 'comparison_3x3_replicas'
title = '$Maze \\ 3x3$'
paths = ['logs/paper/replicas/3x3/qbm/2024-12-17--18-56-22_QRL_QBM',
        #  'logs/paper/replicas/3x3/qbm/2024-12-17--19-15-59_QRL_QBM',
         'logs/paper/replicas/3x3/qbm/2024-12-17--19-19-45_QRL_QBM',
         'logs/paper/replicas/3x3/qbm/2024-12-17--19-24-47_QRL_QBM',
         'logs/paper/replicas/3x3/qbm/2024-12-17--19-34-21_QRL_QBM',
       ]
labels = ['$num. \\ rep. = 1 - binary$', '$num. \\ rep. = 1$', '$num. \\ rep. = 5$', '$num. \\ rep. = 10$']
line_style = ['--', '-', '-', '-', '--']
colors = ['darkslategray', 'darkslategray', 'forestgreen', 'limegreen', 'lightgreen']
r = 0.7
steps = 10_000
names = ['H,', '2,', '5,', '10,']

fig, ax_steps = plt.subplots(1, figsize=(5, 5))

for label, path, color, line in zip(labels, paths, colors, line_style):

    print(path, label)
    folder_path = os.path.basename(os.path.normpath(path))
    results_file_name = "/result.json"
    result_files  = [f.path for f in os.scandir(path) if f.is_dir()] # and name in f.path]
    
    results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]

    curves = []
    curves_x_axis = []
    for i, data_exp in enumerate(results):
        curves.append(data_exp['episode_reward_mean'].values*0.01)
        curves_x_axis.append(data_exp['num_env_steps_sampled'].values)                    
    min_length = min([len(d) for d in curves])
    data = [d[:min_length] for d in curves]
    x_axis = curves_x_axis[0][:min_length]
    data = np.vstack(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    upper = mean +  std
    lower = mean  -  std
    ax_steps.plot(x_axis, mean, label=label, color=color, linestyle=line) #[idx], color=color[idx])
    ax_steps.fill_between(x_axis, lower, upper, alpha=0.5, color=color, linestyle=line) #, color=color[idx])
    # ax_steps.plot(x_axis, mean, label=label, linestyle=line) #[idx], color=color[idx])
    # ax_steps.fill_between(x_axis, lower, upper, alpha=0.5, linestyle=line) #, color=color[idx])


ax_steps.set_xlabel("$environment \\ steps$", fontsize=13)
ax_steps.hlines(y=r, xmin= 0, xmax=steps, linestyles='dashdot', color='black')
ax_steps.set_xlim(0, steps)
# ax_main.axvline(x=2000, label='grover qrl', color='b', linestyle='--')

# Set the axis limits
# ax_steps.set_xlim(0, 20000)
ax_steps.set_ylim(-1., 1.0)


ax_steps.set_ylabel('$reward$', fontsize=15)
# ax_main.set_xlim(0, r)
ax_steps.legend(fontsize=10, loc='lower right')
ax_steps.minorticks_on()
ax_steps.grid(which='both', alpha=0.4)


# plt.text(1.01, 0.6, info_text, transform=ax_main.transAx  es, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
fig.suptitle(title, fontsize=13)

fig.tight_layout()
plt.savefig(f'logs/paper/{fig_name}.png') #, dpi=1200)
print('Done')