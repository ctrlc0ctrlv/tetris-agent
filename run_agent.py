# -*- coding: utf-8 -*-

"""
    RL agent usage
"""

import rl_agent
import gym_tetris


if __name__ == "__main__":
    env = gym_tetris.TetrisEnv()
    state_shape, n_actions = (
        env.state_shape,
        env.n_actions
    )

    agent = rl_agent.DQNAgent(state_shape, n_actions)
