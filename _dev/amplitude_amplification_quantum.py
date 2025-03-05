from copy import deepcopy
from math import ceil

import gymnasium as gym
import numpy as np
import pennylane as qml
from ray import tune


def oracle(action, num_qubits):
    qml.FlipSign(action, wires=range(num_qubits))


def grover_operator(binary_action, num_qubits):
    oracle(binary_action, num_qubits)
    qml.templates.GroverOperator(wires=range(num_qubits))


def quantum_circuit(state, config, grover_circuit_operators, num_qubits):
    # Apply Hadamard to all qubits for superposition

    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    for action in grover_circuit_operators[state]:
        binary_action = bin(action)[2:].zfill(num_qubits)
        action_vector = [int(a) for a in binary_action]
        grover_operator(action_vector, num_qubits)

    # return qml.probs(wires=range(num_qubits))
    return qml.counts(wires=range(num_qubits))


class Grover_agent(tune.Trainable):
    """
    Inits a quantum QLearner object for given environment.
    Environment must be discrete and of "maze type", with the last state as the goal
    """

    def setup(self, config: dict):

        self.config = config["model"]["custom_model_config"]
        # self.env = gym.make("FrozenLake-v1", is_slippery=False) #wrapper_switch[config['env_config']['env']](config['env_config'])
        self.env = wrapper_switch[config["env_config"]["env"]](config["env_config"])

        # state and action spaces dims
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.state_space = self.env.observation_space.shape[0]
        else:
            self.state_space = self.env.observation_space.n

        self.action_space = 4  # self.env.action_space.n
        # dim of qubits register needed to encode all actions
        self.num_qubits = ceil(np.log2(self.action_space))
        # optimal number of steps in original Grover's algorithm
        self.max_grover_steps = int(
            round(np.pi / (4 * np.arcsin(1.0 / np.sqrt(2**self.num_qubits))) - 0.5)
        )
        # quality values
        self.state_vals = np.zeros(self.state_space)
        # grover steps taken
        self.num_grover_steps = np.zeros(
            (self.state_space, self.action_space), dtype=int
        )
        # boolean flags to signal maximum amplitude amplification reached
        self.grover_steps_flag = np.zeros(
            (self.state_space, self.action_space), dtype=bool
        )
        self.grover_circuit_operators = [[] for _ in range(self.state_space)]
        self.device = qml.device("default.qubit", wires=self.action_space, shots=1000)
        self.qnode_actor = qml.QNode(quantum_circuit, self.device, interface="torch")

        self.episodes_trained_total = 0
        self.steps_trained_total = 0
        self.global_step = 0
        self.circuit_executions = 0

    def step(self):
        """
        groverize and measure action qstate -> take corresp action
        obtain: newstate, reward, terminationflag
        update stateval, grover_steps
        for epoch in epochs until max_epochs is reached
        :return:
        dictionary of trajectories
        """

        # set initial max_steps
        num_steps = self.config["steps_per_epoch"]

        done = True

        self.steps_trained_epoch = 0
        self.episodes_trained_epoch = 0
        self.rewards_epoch = []
        self.policy_loss_epoch = []
        self.env_steps_per_epoch = []

        rewards_epoch_train = []
        log_probs_epoch_train = []
        steps_per_episode = 0
        rewards_episode = []
        log_probs_episode = []

        # reset env
        state = self.env.reset()[0]
        # init list for traj
        traj = [state]

        # alternative steps pseudo code:
        # 1. find max action
        # max_key = max(probs, key=probs.get)
        # a_max = int(max_key, 2)
        # 2. compute ratio
        # phi =
        # 3.
        # a = 1.3
        # b = 15.8
        # c = 0.65
        # phi_1 = np.pi * (a * phi + b)
        # phi_2 = c * phi_1
        for step in range(num_steps):

            if self.config["mode"] == "grover":
                probs = self.qnode_actor(
                    state, self.config, self.grover_circuit_operators, self.num_qubits
                )
                self.circuit_executions += 1
                max_key = max(probs, key=probs.get)
                action = int(max_key, 2)
            elif self.config["mode"] == "softmax":
                pass
            # take action
            new_state, reward, done, _, _ = self.env.step(action)

            rewards_episode.append(reward)

            steps_per_episode += 1
            self.steps_trained_epoch += 1
            self.steps_trained_total += 1
            # print(new_state, done)
            # if new_state == state:
            #     reward -= 10
            #     done = True

            # if new_state == 0:
            #     reward += 99
            #     # print('Reward #######################')
            # elif not done:
            #     reward -= 1
            reward = reward * 100
            # update statevals and grover steps
            self.state_vals[state] += self.config["alpha"] * (
                reward
                + self.config["gamma"] * self.state_vals[new_state]
                - self.state_vals[state]
            )

            steps_num = int(self.config["k"] * (reward + self.state_vals[new_state]))
            self.num_grover_steps[state, action] = min(steps_num, self.max_grover_steps)

            flag = self.grover_steps_flag[state, :]
            gsteps = self.num_grover_steps[state, action]

            if not flag.any():
                for _ in range(gsteps):
                    self.grover_circuit_operators[state].append(action)

            if gsteps >= self.max_grover_steps and not flag.any():
                self.grover_steps_flag[state, action] = True

            state = new_state

            # quit epoch if done
            if done or (step == num_steps - 1):
                self.rewards_epoch.append(np.sum(rewards_episode))
                self.env_steps_per_epoch.append(steps_per_episode)
                self.episodes_trained_total += 1
                self.episodes_trained_epoch += 1
                steps_per_episode = 0
                # print('done:', done, step)
                break

        # return trajectories
        return deepcopy(
            {
                "steps_trained_total": self.steps_trained_total,
                "circuit_executions": self.circuit_executions,
                "episode_reward_mean": np.mean(self.rewards_epoch),
                "episode_reward_max": np.max(self.rewards_epoch),
                "episode_reward_min": np.min(self.rewards_epoch),
                "env_steps_per_epoch": self.env_steps_per_epoch,
                "num_env_steps_sampled": self.steps_trained_total,
            }
        )

    # 	"episode_length_mean": np.mean(self.env_steps_per_epoch),
    # 	"mean_env_steps_per_episode": np.mean(self.env_steps_per_epoch),
    # #  "policy_loss_epoch": policy_loss_episode,
    # 	"steps_trained_epoch": self.steps_trained_epoch,
    # 	"episodes_trained_total": self.episodes_trained_total,
    # 	"episodes_trained_epoch": self.episodes_trained_epoch,
    # 	})
