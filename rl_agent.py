# -*- coding: utf-8 -*-

"""
    RL agent for tetris gym environment
"""


import numpy as np
import torch
from torch import nn


class DQNAgent(nn.Module):
    """
        Agent class
    """

    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        assert len(state_shape) == 1
        state_dim = state_shape[0]
        hidden_size_1 = 100
        hidden_size_2 = 50
        self._nn = nn.Sequential(
            nn.Linear(state_dim, hidden_size_1),
            nn.ReLU(),
            # nn.BatchNorm1d(200),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, n_actions),
            nn.ReLU(),
        )

    def forward(self, state_t):
        """
            Takes agent's observation (tensor), returns qvalues (tensor)

            Args:
                state_t (_type_): a batch states,
                shape = [batch_size, *state_dim=200]
        """
        # Use your network to compute qvalues for given state
        qvalues = self._nn(state_t)
        # print(f"qvalues shape: {qvalues.shape}")

        assert (
            qvalues.requires_grad
        ), "qvalues must be a torch tensor with grad"
        assert (
            len(qvalues.shape) == 2
            and qvalues.shape[0] == state_t.shape[0]
            and qvalues.shape[1] == self.n_actions
        )

        return qvalues

    def get_qvalues(self, states):
        """
            Like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """
            Pick actions given qvalues.
            Uses epsilon-greedy exploration strategy
        """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon]
        )
        return np.where(should_explore, random_actions, best_actions)


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """
        Plays n_games full games. If greedy, picks actions as argmax(qvalues).
        Returns mean reward
    """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues(np.asarray([s]))
            action = (
                qvalues.argmax(axis=-1)[0]
                if greedy
                else agent.sample_actions(qvalues)[0]
            )
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    rewards = np.array([rewards])
    return np.mean(rewards)
