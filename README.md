# CleanQRL (Clean Quantum Reinforcement Learning)



[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/georgkruse/cleanqrl?tab=License-1-ov-file)
[![docs](https://img.shields.io/github/deployments/vwxyzjn/cleanrl/Production?label=docs&logo=vercel)](https://georgkruse.github.io/cleanqrl-docs/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

**CleanQRL** is a Reinforcement Learning library specifically tailored to the subbranch of Quantum Reinforcement Learning and is greatly inspired by the amazing work of **[CleanRL](https://github.com/vwxyzjn/cleanrl)**. Just as the classical analougue, we aim to provide high-quality single-file implementation with research-friendly features. The implementation follows mainly the ideas of **[CleanRL](https://github.com/vwxyzjn/cleanrl)** and is clean and simple, yet can scale nicely trough additional features such as **[ray tune](https://docs.ray.io/en/latest/tune/index.html)**. The main features of this repository are


* 📜 Single-file implementations of classical and quantum version of 5+ Reinforcement Learning agents 
* 💾 Tuned and Benchmarked agents (with available configs)
* 🎮 Integration of gymnasium, mujoco and jumanji
* 📘 Examples on how to enhance the standard QRL agents on a variety of games
* 📈 Tensorboard Logging
* 🌱 Local Reproducibility via Seeding
* 🧫 Experiment Management with [Weights and Biases](https://wandb.ai/site)
* 📊 Easy and straight forward hyperparameter tuning with [ray tune](https://docs.ray.io/en/latest/tune/index.html)

What we are missing compared to **[CleanRL](https://github.com/vwxyzjn/cleanrl)**:

* 💸 Cloud Integration with docker and AWS 
* 📹 Videos of Gameplay Capturing


You can read more about CleanRL in [our upcoming paper]() and [see the docs for additional documentation](https://georgkruse.github.io/cleanqrl-docs/).



> ⚠️ **NOTE**: CleanQRL is greatly based on CleanRL. The implementations of the classical agents are entirely based on these implementations, but updated to gymnasium 1.0 as well as adapted to ray tune. The quantum versions of the algorithms are done with as little changes to the classical versions as possible, such that it is easy to understand.

> ⚠️ **NOTE**: This repository is meant for people interested in QRL and want to get started as quick as possible. However, the algorithms are not meant as imports as popular other RL libraries such as RLlib and StableBaselines. Instead, our implementations aims to make it easy to understand the algorithms and enable users to quickly adapt these implementations to their needs with easy debugging possible. 

## Get started

To run experiments locally, give the following a try:

```bash
git clone https://github.com/georgkruse/cleanqrl.git
cd cleanqrl
conda env create -f environment.yaml
```

## Run first experiments

Each agent can be run as a single file, either from the parent directory or directly in the subfolder. 

```
python cleanrl/dqn_quantum.py 
```

```
cd cleanqrl 
python ppo_classical.py 
```

The parameters can be changed in the ```Config``` class at the end of each file:

```python
@dataclass
class Config:
    # General parameters
    trial_name: str = 'reinforce_quantum'  # Name of the trial
    trial_path: str = 'logs'  # Path to save logs relative to the parent directory
    wandb: bool = True # Use wandb to log experiment data 

    # Environment parameters
    env_id: str = "CartPole-v1" # Environment ID
    
    # Algorithm parameters
    num_envs: int = 1  # Number of environments
    total_timesteps: int = 100000  # Total number of timesteps
    gamma: float = 0.99  # discount factor
    lr_input_scaling: float = 0.01  # Learning rate for input scaling
    lr_weights: float = 0.01  # Learning rate for variational parameters
    lr_output_scaling: float = 0.01  # Learning rate for output scaling
    cuda: bool = False  # Whether to use CUDA
    num_qubits: int = 4  # Number of qubits
    num_layers: int = 2  # Number of layers in the quantum circuit
    device: str = "default.qubit"  # Quantum device
    diff_method: str = "backprop"  # Differentiation method
    save_model: bool = True # Save the model after the run

```

Additionally, all algorithms can be executed using the ```main functions``` in the root directory. All parameters for the algorithms are specified in the ```configs``` folder. [See additional information in the documentation.]()

By default, all metrics are logged to a json file. 

For tensorboard logging, you can use  ray tune (see more in the following section and the [docs]())

To use experiment tracking with [wandb](), you need to set the boolean variable ```wandb``` in the config class or the config file to ```True``` and then run: 

```bash
wandb login # only required for the first time
python cleanrl/ppo_quantum.py \
```
## Hyperparameter Tuning 



## Algorithms Implemented



| Algorithm      | Variants Implemented |
| ----------- | ----------- |
| ✅ [REINFORCE]() |  [`reinforce_classical.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/reinforce_classical.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#reinforce_classicalpy) |
| | [`reinforce_quantum.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/reinforce_quantum.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#reinforce_quantumpy) |
| ✅ [Deep Q-Learning (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) | [`dqn_classical.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_classical.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#dqn_classicalpy) |
| | [`dqn_quantum.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#dqn_quantumpy) |
| ✅ [Proximal Policy Gradient (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  |  [`ppo_classical.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_classicalpy) |
| |  [`ppo_classical_continuous.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical_continuous.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_classical_continuouspy) |
| |  [`ppo_classical_jumanji.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical_jumanji.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_classical_jumanjipy) |
| |  [`ppo_quantum.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_quantumpy) |
| |  [`ppo_quantum_continuous.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_continuous.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_quantum_continuouspy) |
| |  [`ppo_quantum_jumanji.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_jumanji.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_quantum_jumanjipy) |
| ✅ [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) |  [`ddpg_classical.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ddpg_classical.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ddpg_classicalpy) |
| | [`ddpg_quantum.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ddpg_quantum.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ddpg_quantumpy) |

## Open (Q)RL Benchmark

To make our experimental data transparent, CleanQRL participates in a related project called [Open RL Benchmark](https://github.com/openrlbenchmark/openrlbenchmark), which contains tracked experiments from popular DRL libraries such as ours, [Stable-baselines3](https://github.com/DLR-RM/stable-baselines3), [openai/baselines](https://github.com/openai/baselines), [jaxrl](https://github.com/ikostrikov/jaxrl), and others. 

Since the field of QRL is relatively new, we are not aware of any benchmarks of 
## Support and Community

We want to grow as a community, so posting [Github Issues](https://github.com/vwxyzjn/cleanrl/issues) and PRs are very welcome! If you are missing and algorithms or have a specific problem to which you want to tailor your QRL algorithms but fail to do so, you can also create a feature request!

## Citing CleanQRL

If you use CleanQRL in your work, please cite our technical [paper]():

```bibtex

```


## Other works which CleanQRL implementations are based on


## Other works using CleanQRl

