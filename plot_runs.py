import yaml
import os
import ray
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from glob import glob
import numpy as np
import re

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


fig, axis = plt.subplots(figsize=(8,7))

root_path = '/home/users/coelho/ray_results/train_2024-12-13_11-38-34'
paths, labels, types = [], [], []
paths += [root_path]

title = '4 qubits local grid 1 env'
fig_name = f'{title}'
data_length = 300


labels += [["0.1",
            "0.05",
            "0.01",
            "0.005",
            "0.001"
        ]]

types += [{'0': ['=0.1'],
           '1': ['=0.05'],
           '2': ['=0.01'],
           '3': ['=0.005'],
           '4': ['=0.001']
        }]

for idx, path in enumerate(paths):
    curves = [[] for _ in range(len(labels[idx]))]
    curves_x = [[] for _ in range(len(labels[idx]))]


    results_file_name = "/result.json"
    result_files  = [f.path for f in os.scandir(path) if f.is_dir() ]
    results = [pd.read_json(f + results_file_name,lines=True) for f in result_files]
    result = pd.concat(results)
    
    for key, value in types[idx].items():
        for i, data_exp in enumerate(results):
            if all(x in result_files[i] for x in value):
                curves_x[int(key)].append(data_exp['episode'].values[:data_length])
                curves[int(key)].append(data_exp['episodic_return'].values[:data_length])


    for id, curve in enumerate(curves):
        data = np.vstack(curve)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        upper = mean +  std
        lower = mean  -  std
        axis.plot(data_exp['episode'].values[:data_length], mean, label=labels[idx][id])
        axis.fill_between(data_exp['episode'].values[:data_length], lower, upper, alpha=0.5)



axis.set_xlabel("$Episodes$", fontsize=13)
axis.set_ylabel("$Return$", fontsize=15)
axis.set_title(title, fontsize=15)
axis.legend(fontsize=12, loc='lower left')
axis.minorticks_on()
axis.grid(which='both', alpha=0.4)
fig.tight_layout()
fig.savefig(f'{fig_name}.png', dpi=100)
