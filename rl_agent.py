# -*- coding: utf-8 -*-

"""
    RL agent for tetris gym environment
"""


import numpy as np
import torch
from torch import nn
import logging
from gym.wrappers.monitoring import video_recorder, stats_recorder
import gym_tetris


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
            # nn.InstanceNorm1d(hidden_size_1),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            # nn.InstanceNorm1d(hidden_size_2),
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
        states = torch.tensor(
            np.asarray(states), device=model_device, dtype=torch.float32
        )
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


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps,
    record every (s,a,r,s', done) to replay buffer.

    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])

        action = agent.sample_actions(qvalues)[0]
        # action = action.argmax(axis=-1)[0]
        state, reward, done, _ = env.step(action)
        sum_rewards += reward

        exp_replay.add(s, action, reward, state, done)

        if done:
            state = env.reset()
        s = state

    return sum_rewards, s


def compute_td_loss(
    states,
    actions,
    rewards,
    next_states,
    is_done,
    agent,
    target_network,
    gamma=0.99,
    check_shapes=False,
    device=torch.device("cpu"),
):
    """
        Compute td loss using torch operations only. Use the formulae above
    """
    states = torch.tensor(
        states, device=device, dtype=torch.float32
    )  # shape: [batch_size, *state_shape]
    actions = torch.tensor(
        actions, device=device, dtype=torch.int64
    )  # shape: [batch_size]
    rewards = torch.tensor(
        rewards, device=device, dtype=torch.float32
    )  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype("float32"), device=device, dtype=torch.float32,
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]

    # compute q-values for all actions in next states
    # with torch.no_grad():
    predicted_next_qvalues = target_network(
        next_states
    )  # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[
        range(len(actions)), actions
    ]  # shape: [batch_size]

    # compute V*(next_states) using predicted next q-values
    # next_state_values = <YOUR CODE>
    next_state_values = predicted_next_qvalues.max(axis=-1)[0]

    assert (
        next_state_values.dim() == 1
        and next_state_values.shape[0] == states.shape[0]
    ), "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    # target_qvalues_for_actions = <YOUR CODE>
    target_qvalues_for_actions = (
        rewards + gamma * next_state_values * is_not_done
    )

    # target_qvalues_for_actions = torch.where(is_done, rewards, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = torch.mean(
        (predicted_qvalues_for_actions - target_qvalues_for_actions.detach())
        ** 2
    )

    if check_shapes:
        assert (
            predicted_next_qvalues.data.dim() == 2
        ), "make sure you predicted q-values for all actions in next state"
        assert (
            next_state_values.data.dim() == 1
        ), "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert (
            target_qvalues_for_actions.data.dim() == 1
        ), "there's something wrong with target q-values, they must be a vector"

    return loss


def test_and_record_video(
    agent, out_dir="./../../random-agent-results", greedy=True
):
    # logger setup
    logging.basicConfig()

    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().

    # env = gym.make("Tetris-v0", state_mode="matrix")
    env = gym_tetris.TetrisEnv(state_mode="matrix")
    env.reset()
    # env = monitor.Monitor(env, directory=out_dir, force=True)
    vid = video_recorder.VideoRecorder(
        env=env, path=out_dir + "/tetris.mp4", enabled=True
    )
    stats = stats_recorder.StatsRecorder(
        directory=out_dir, file_prefix="tetris",
    )

    EPISODE_COUNT = 100
    MAX_STEPS = 200
    reward = 0
    done = False

    for i in range(EPISODE_COUNT):
        stats.before_reset()
        ob = env.reset()
        # print(ob)
        stats.after_reset(ob)

        # for j in range(MAX_STEPS):
        while True:
            qvalues = agent.get_qvalues(np.asarray([ob]))
            action = (
                qvalues.argmax(axis=-1)[0]
                if greedy
                else agent.sample_actions(qvalues)[0]
            )
            stats.before_step(action)
            ob, reward, done, info = env.step(action)
            vid.capture_frame()
            # print(ob)
            stats.after_step(
                observation=ob, reward=reward, done=done, info=info
            )
            if done:
                # ob = env.reset()
                break
            # Note there's no env.render() here.
            # But the environment still can open window and render
            # if asked by env.monitor: it calls env.render('rgb_array')
            # to record video.
            # Video is not recorded every episode,
            # see capped_cubic_video_schedule for details.
        stats.save_complete()

    # Dump result info to disk
    vid.close()
    stats.close()
    env.close()

    logger.info("Successfully ran Agent.")
    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    # gym.gym.upload(out_dir)
