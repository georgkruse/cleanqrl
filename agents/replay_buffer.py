import numpy as np
import random

'''
Replay Buffer adapted from StableBaselines3
'''

class ReplayBuffer(object):
    def __init__(self, size: int):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    @staticmethod
    def _normalize_obs(obs: np.ndarray,
                       env = None) -> np.ndarray:
        """
        Helper for normalizing the observation.
        """
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env = None) -> np.ndarray:
        """
        Helper for normalizing the reward.
        """
        if env is not None:
            return env.normalize_reward(reward)
        return reward

    def _encode_sample(self, idxes, env = None):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(obs_t)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        return (self._normalize_obs(obses_t, env),
                np.array(actions),
                self._normalize_reward(np.array(rewards), env),
                self._normalize_obs(obses_tp1, env),
                np.array(dones))

    def sample(self, batch_size: int, env = None):

        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, env=env)



class ReplayBufferQBM(object):
    def __init__(self, size: int):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size
    # state, action, reward, next_state, done, free_energy, samples, visible_nodes
    def add(self, obs_t, action, reward, obs_tp1, done, free_energy, samples, visible_nodes):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done, free_energy, samples, visible_nodes)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, obs_tp1, done, free_energy, samples, visible_nodes):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done, free_energy, samples, visible_nodes):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    @staticmethod
    def _normalize_obs(obs: np.ndarray,
                       env = None) -> np.ndarray:
        """
        Helper for normalizing the observation.
        """
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env = None) -> np.ndarray:
        """
        Helper for normalizing the reward.
        """
        if env is not None:
            return env.normalize_reward(reward)
        return reward

    def _encode_sample(self, idxes, env = None):
        obses_t, actions, rewards, obses_tp1, dones, free_energy, samples, visible_nodes = [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, free_energy, samples, visible_nodes = data
            obses_t.append(obs_t)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        return obs_t, action, reward, obs_tp1, done, free_energy, samples, visible_nodes

    def sample(self, batch_size: int, env = None):

        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, env=env)
