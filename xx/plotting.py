import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import csv


class Plotter():
    """
    This class will be used to plot the results from the runs.
    """

    def __init__(self,run_path):
        self.run_path = run_path

    def load_run(self, gradients = True):
        all_dirs = os.listdir(self.run_path)
        self.runs = [dir for dir in all_dirs if dir.startswith("get_gradient")]

        if gradients:
            results = []
            for run in self.runs:
                aux = []
                result_file_path = os.path.join(self.run_path, run, 'result.json')
                with open(result_file_path, 'r') as file:
                    for line in file:
                        try:
                            aux.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                    results.append(aux)
        else:
            results = []
            for run in self.runs:
                result_file_path = os.path.join(self.run_path, run, 'result.json')
                if os.path.isfile(result_file_path):
                    with open(result_file_path, 'r') as file:
                        results.append(json.load(file))
        
        params = []
        for run in self.runs:
            params_file_path = os.path.join(self.run_path, run, 'params.json')
            if os.path.isfile(params_file_path):
                with open(params_file_path, 'r') as file:
                    params.append(json.load(file))
        
        self.results = results
        self.params = params


    def gradient_plots(self,params,save_path, figure_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_plot = []
        for i, param in enumerate(params):
            aux = []
            for j,run in enumerate(self.runs):
                if param in run:
                    aux.append(self.results[j])
            results_plot.append(aux)

        # Get initial and final KTA, train_acc, test_acc
        results_gathered_input_scaling = []
        results_gathered_variational = []
        for result in results_plot:
            aux_input_scaling = []
            aux_variational = []
            for sub_result in result:
                for sample in sub_result:
                    aux_input_scaling.append([sample["gradients"]["input_scaling_actor"]])
                    aux_variational.append([sample["gradients"]["variational_actor"]])
            results_gathered_input_scaling.append(np.array(aux_input_scaling))
            results_gathered_variational.append(np.array(aux_variational))

        vars_variational = []
        vars_input_scaling = []
        for i in range(len(results_gathered_input_scaling)):
            run_input_scaling = np.array(results_gathered_input_scaling[i])
            run_variational = np.array(results_gathered_variational[i])
            run_input_scaling = run_input_scaling.reshape(run_input_scaling.shape[0],run_input_scaling.shape[2])
            run_variational = run_variational.reshape(run_variational.shape[0],run_variational.shape[2])
            mean_input_scaling = np.mean(run_input_scaling, axis = 1)
            mean_variational = np.mean(run_variational, axis = 1)
            var_input_scaling = np.var(mean_input_scaling)
            var_variational = np.var(mean_variational)
            vars_variational.append(var_variational)
            vars_input_scaling.append(var_input_scaling)
        fig, axs = plt.subplots(1,2,figsize=(8,4), tight_layout=True)
        axs[0].scatter([2,4,6,8,10,12], vars_input_scaling)
        axs[0].set_xlabel("Number of Qubits")
        axs[0].set_ylabel("Variance of Gradient")
        axs[0].set_yscale("log")
        axs[0].set_title("Input Scaling")
        axs[1].scatter([2,4,6,8,10,12], vars_variational)
        axs[1].set_xlabel("Number of Qubits")
        axs[1].set_ylabel("Variance of Gradient")
        axs[1].set_yscale("log")
        axs[1].set_title("Variational")
        plt.savefig(os.path.join(save_path, f"{figure_name}.png"), dpi = 300)

    def gradient_plots_v2(self,params,save_path, figure_name, labels):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        fig, axs = plt.subplots(1,2,figsize=(8,4), tight_layout=True)
        axs[0].set_xlabel("Number of Qubits")
        axs[0].set_ylabel("Variance of Gradient")
        axs[0].set_yscale("log")
        axs[0].set_title("Input Scaling")
        axs[1].set_xlabel("Number of Qubits")
        axs[1].set_ylabel("Variance of Gradient")
        axs[1].set_yscale("log")
        axs[1].set_title("Variational Parameters")

        label = labels[0]

        for idx, param in enumerate(params):
            results_plot = []
            for i, p in enumerate(param):
                aux = []
                for j,run in enumerate(self.runs):
                    if p in run:
                        aux.append(self.results[j])
                results_plot.append(aux)

            # Get initial and final KTA, train_acc, test_acc
            results_gathered_input_scaling = []
            results_gathered_variational = []
            for result in results_plot:
                aux_input_scaling = []
                aux_variational = []
                for sub_result in result:
                    for sample in sub_result:
                        aux_input_scaling.append([sample["gradients"]["input_scaling_actor"]])
                        aux_variational.append([sample["gradients"]["variational_actor"]])
                results_gathered_input_scaling.append(np.array(aux_input_scaling))
                results_gathered_variational.append(np.array(aux_variational))

            vars_variational = []
            vars_input_scaling = []
            for i in range(len(results_gathered_input_scaling)):
                run_input_scaling = np.array(results_gathered_input_scaling[i])
                run_variational = np.array(results_gathered_variational[i])
                run_input_scaling = run_input_scaling.reshape(run_input_scaling.shape[0],run_input_scaling.shape[2])
                run_variational = run_variational.reshape(run_variational.shape[0],run_variational.shape[2])
                mean_input_scaling = np.mean(run_input_scaling, axis = 1)
                mean_variational = np.mean(run_variational, axis = 1)
                var_input_scaling = np.var(mean_input_scaling)
                var_variational = np.var(mean_variational)
                vars_variational.append(var_variational)
                vars_input_scaling.append(var_input_scaling)

            axs[0].scatter([2,4,6,8,10,12], vars_input_scaling, label = label[idx])
            axs[1].scatter([2,4,6,8,10,12], vars_variational, label = label[idx])
            
        plt.legend()
        plt.savefig(os.path.join(save_path, f"{figure_name}.png"), dpi = 300)

        
if __name__ == "__main__":
    plotter = Plotter("/home/users/coelho/ray_results/get_gradients_2024-11-29_17-12-07")
    plotter.load_run(gradients = True)

    params_0 = [
        "uniform,num_qubits=2",
        "uniform,num_qubits=4",
        "uniform,num_qubits=6",
        "uniform,num_qubits=8",
        "uniform,num_qubits=10",
        "uniform,num_qubits=12",
    ]

    params_1 = [
        "reduced_domain_fixed,num_qubits=2",
        "reduced_domain_fixed,num_qubits=4",
        "reduced_domain_fixed,num_qubits=6",
        "reduced_domain_fixed,num_qubits=8",
        "reduced_domain_fixed,num_qubits=10",
        "reduced_domain_fixed,num_qubits=12",
    ]

    params_2 = [
        "reduced_domain,num_qubits=2",
        "reduced_domain,num_qubits=4",
        "reduced_domain,num_qubits=6",
        "reduced_domain,num_qubits=8",
        "reduced_domain,num_qubits=10",
        "reduced_domain,num_qubits=12",
    ]

    params_3 = [
        "small_random,num_qubits=2",
        "small_random,num_qubits=4",
        "small_random,num_qubits=6",
        "small_random,num_qubits=8",
        "small_random,num_qubits=10",
        "small_random,num_qubits=12",
    ]


    labels_0 = [
        "uniform",
        "reduced_domain_fixed",
        "reduced_domain",
        "small_random"
    ]

    labels = [
        labels_0
    ]


    params = [
        params_0, params_1, params_2, params_3
    ]

    figure_names = ["gradients"]

    plotter.gradient_plots_v2(params = params, save_path = "figures/gradients/",figure_name = figure_names[0], labels = labels)

    #for i in range(len(params)):
    #    plotter.gradient_plots(params = params[i], save_path = "figures/gradients/",figure_name = figure_names[i])