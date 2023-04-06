import numpy as np
from ma_gym.envs.combat.combat import Combat
from MADDPG import MADDPG
import rl_utils
from torch.utils.tensorboard import SummaryWriter
import torch as T
import matplotlib.pyplot as plt
import time

def to_one_hot(label, dimension=12):
    results = np.zeros((len(label), dimension))
    for i, label in enumerate(label):
        results[i, label] = 1.
    return results
if __name__=='__main__':
    writer = SummaryWriter('runs/maddpg')
    scenario = 'simple'
    n_agents = 9
    grid_size = (16, 16)
    env = Combat(grid_shape=grid_size, n_agents=n_agents, n_opponents=n_agents)
    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    update_interval = 100
    minimal_size = 4000
    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])
    critic_input_dim = sum(state_dims) + sum(action_dims)
    maddpg = MADDPG(n_agents, T.device("cuda:0" if (T.cuda.is_available()) else "cpu"), actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dim, gamma, tau)
    PRINT_INTEMVAL = 200
    num_episodes = 10000
    MAX_STEPS = 50
    total_steps = 0
    score_history = []
    best_score = 0
    evaluate = True
    return_list = []  # 记录每一轮的回报（return）
    total_step = 0
    if evaluate:
        state = env.reset()
        maddpg.load_checkpoint()
        done = [False] * n_agents
        score=0
        while not any(done):
            env.render()
            time.sleep(0.5)
            actions = maddpg.take_action(state, explore=True)
            Actions = np.argmax(actions, axis=1)
            next_state, reward, done, _ = env.step(Actions)
            state = next_state
            print(Actions)
            print(reward)
            score+=sum(reward)
        print(env.agent_health)
        print(score)