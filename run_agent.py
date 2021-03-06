# -*- coding: utf-8 -*-

"""
    RL agent usage
"""

from rl_agent import (
    play_and_record,
    DQNAgent,
    compute_td_loss,
    evaluate,
    test_and_record_video,
)
import gym_tetris
import replay_buffer
import torch
import torch.nn as nn
import psutil
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, gaussian
from tqdm import trange
from IPython.display import clear_output
import numpy as np
import gym


def test_code(agent, env):
    # testing your code.
    exp_replay = replay_buffer.ReplayBuffer(2000)

    state = env.reset()
    play_and_record(state, agent, env, exp_replay, n_steps=1000)

    # if you're using your own experience replay buffer,
    # some of those tests may need correction.
    # just make sure you know what your code does
    assert len(exp_replay) == 1000, (
        "play_and_record should have added exactly 1000 steps, "
        "but instead added %i" % len(exp_replay)
    )
    is_dones = list(zip(*exp_replay._storage))[-1]

    # assert 0 < np.mean(is_dones) < 0.1, \
    # "Please make sure you restart the game whenever it is 'done' and " \
    # "record the is_done correctly into the buffer. Got %f is_done rate over"
    # "%i steps. [If you think it's your tough luck, just re-run the test]" % (
    # np.mean(is_dones), len(exp_replay))

    for _ in range(100):
        (
            obs_batch,
            act_batch,
            reward_batch,
            next_obs_batch,
            is_done_batch,
        ) = exp_replay.sample(10)
        # print(obs_batch.shape, next_obs_batch.shape, (10,) + state_shape)
        assert obs_batch.shape == next_obs_batch.shape == (10,) + state_shape
        assert act_batch.shape == (10,), (
            "actions batch should have shape (10,) but is instead %s"
            % str(act_batch.shape)
        )
        assert reward_batch.shape == (10,), (
            "rewards batch should have shape (10,) but is instead %s"
            % str(reward_batch.shape)
        )
        assert is_done_batch.shape == (10,), (
            "is_done batch should have shape (10,) but is instead %s"
            % str(is_done_batch.shape)
        )
        assert [
            int(i) in (0, 1) for i in is_dones
        ], "is_done should be strictly True or False"
        assert [
            0 <= a < n_actions for a in act_batch
        ], "actions should be within [0, n_actions)"

    print("Test success!")


def is_enough_ram(min_available_gb=0.1):
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024 ** 3)


def fill_replay_buffer(env, agent, size):
    exp_replay = replay_buffer.ReplayBuffer(size)
    for _ in range(100):
        if not is_enough_ram(min_available_gb=0.1):
            print(
                """
                Less than 100 Mb RAM available.
                Make sure the buffer size in not too huge.
                Also check, maybe other processes consume RAM heavily.
                """
            )
            break
        play_and_record(state, agent, env, exp_replay, n_steps=1000)
        if len(exp_replay) == size:
            break
    print("Buffer filled success!")
    return exp_replay


def make_env(seed=0):
    env = gym_tetris.TetrisEnv()
    env.seed = seed
    return env


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (
        init_val * (total_steps - cur_step) + final_val * cur_step
    ) / total_steps


def smoothen(values):
    kernel = gaussian(100, std=100)
    # kernel = np.concatenate([np.arange(100), np.arange(99, -1, -1)])
    kernel = kernel / np.sum(kernel)
    return fftconvolve(values, kernel, "valid")


def train(agent, env):
    # learning preparation
    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = 3 * 10 ** 4
    decay_steps = 1 * 10 ** 3
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
    init_epsilon = 1
    final_epsilon = 0.1
    loss_freq = 20
    refresh_target_network_freq = 100
    eval_freq = 1000
    max_grad_norm = 5000
    mean_rw_history = []
    td_loss_history = []
    grad_norm_history = []
    initial_state_v_history = []
    step = 0

    state = env.reset()
    with trange(step, total_steps + 1) as progress_bar:
        for step in progress_bar:
            if not is_enough_ram():
                print("less that 100 Mb RAM available, freezing")
                # print(
                # "make sure everything is ok and
                # use KeyboardInterrupt to continue"
                # )
                # wait_for_keyboard_interrupt()
                break

            agent.epsilon = linear_decay(
                init_epsilon, final_epsilon, step, decay_steps
            )

            # play
            _, state = play_and_record(
                state, agent, env, exp_replay, timesteps_per_epoch
            )

            # train
            # <YOUR CODE: sample batch_size of data from experience replay>
            s, a, r, next_s, is_done = exp_replay.sample(batch_size)
            print(s.shape)
            # loss = <YOUR CODE: compute TD loss>
            loss = compute_td_loss(
                s, a, r, next_s, is_done, agent, target_network
            )

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                agent.parameters(), max_grad_norm
            )
            opt.step()
            opt.zero_grad()

            if step % loss_freq == 0:
                td_loss_history.append(loss.data.cpu().item())
                grad_norm_history.append(grad_norm)

            if step % refresh_target_network_freq == 0:
                # Load agent weights into target_network
                # <YOUR CODE>
                target_network.load_state_dict(agent.state_dict())

            if step % eval_freq == 0:
                mean_rw_history.append(
                    evaluate(
                        make_env(seed=step),
                        agent,
                        n_games=10,
                        greedy=True,
                        t_max=1000,
                    )
                )
                initial_state_q_values = agent.get_qvalues(
                    [make_env(seed=step).reset()]
                )
                initial_state_v_history.append(np.max(initial_state_q_values))

                clear_output(True)
                # print(
                # "buffer size = %i, epsilon = %.5f"
                # % (len(exp_replay), agent.epsilon)
                # )

                plt.figure(figsize=[16, 9])

                plt.subplot(2, 2, 1)
                plt.title("Mean reward per episode")
                plt.plot(mean_rw_history)
                plt.grid()

                assert not np.isnan(td_loss_history[-1])
                plt.subplot(2, 2, 2)
                plt.title("TD loss history (smoothened)")
                plt.plot(smoothen(td_loss_history))
                plt.grid()

                plt.subplot(2, 2, 3)
                plt.title("Initial state V")
                plt.plot(initial_state_v_history)
                plt.grid()

                plt.subplot(2, 2, 4)
                plt.title("Grad norm history (smoothened)")
                plt.plot(smoothen(grad_norm_history))
                plt.grid()

                plt.savefig(str(step) + ".png")
                plt.close()
                # plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # now device == "cpu"

    # preparation
    SEED = 2
    # env = gym_tetris.TetrisEnv()
    env = gym.make("Tetris-v0")
    env.seed = SEED
    state_shape, n_actions = (env.state_shape, env.n_actions)
    state = env.reset()
    agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
    target_network = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())

    env.spec.max_episode_steps = 200

    # video time!
    test_and_record_video(agent=agent, out_dir="./results")

    # testing everything
    test_code(agent=agent, env=env)

    # filling replay buffer
    REPLAY_BUFFER_SIZE = 10 ** 4
    exp_replay = fill_replay_buffer(
        agent=agent, env=env, size=REPLAY_BUFFER_SIZE
    )

    train(agent=agent, env=env)

    # video time!
    test_and_record_video(agent=target_network, out_dir="./results_learned")
