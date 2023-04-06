import numpy as np
from ma_gym.envs.combat.combat import Combat
from MADDPG import MADDPG
import rl_utils
from torch.utils.tensorboard import SummaryWriter
import torch as T
import matplotlib.pyplot as plt
import time



def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state
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
    replay_buffer = rl_utils.ReplayBuffer(100000)
    PRINT_INTEMVAL = 200
    num_episodes = 20000
    MAX_STEPS = 50
    score_history = []
    best_score = 0
    evaluate = False
    return_list = []  # 记录每一轮的回报（return）
    total_step = 0

    for i_episode in range(num_episodes):
        state = env.reset()
        score = 0
        score1 = 0
        episode_step = 0
        # ep_returns = np.zeros(len(env.agents))
        done = [False] * n_agents
        if i_episode < 10:
            while not any(done):
                env.render()
                actions = env.choose_action()

                next_state, reward, done, _ = env.step(actions)
                score1 += sum(reward)
                actions = to_one_hot(actions, 12)
                replay_buffer.add(state, actions, reward, next_state, done)
                state = next_state

                #time.sleep(2)
            print('episode', i_episode, 'average score {:.1f}'.format(score1))
        else:

            while not any(done):
                env.render()
                actions = maddpg.take_action(state, explore=True)
                Actions = np.argmax(actions, axis=1)
                next_state, reward, done, _ = env.step(Actions)

                score += sum(reward)
                replay_buffer.add(state, actions, reward, next_state, done)
                state = next_state
                total_step += 1
                if replay_buffer.size(
                ) >= 4000 and total_step % update_interval == 0:
                    sample = replay_buffer.sample(1024)


                    def stack_array(x):
                        rearranged = [[sub_x[i] for sub_x in x]
                                      for i in range(len(x[0]))]
                        return [
                            T.FloatTensor(np.vstack(aa)).to("cuda:0" if (T.cuda.is_available()) else "cpu")
                            for aa in rearranged
                        ]


                    sample = [stack_array(x) for x in sample]
                    for a_i in range(n_agents):
                        maddpg.update(sample, a_i)
                    maddpg.update_all_targets()
                episode_step += 1
                if episode_step > MAX_STEPS:
                    done = [True] * n_agents

            maddpg.step()
            writer.add_scalar('step_rewards_{}'.format('maddpg'), score, global_step=i_episode)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if not evaluate:
                if avg_score > best_score:
                    maddpg.save_checkpoint()
                    best_score = avg_score
            if i_episode % 100 == 0 and i_episode > 0:
                print('episode', i_episode, 'average score {:.1f}'.format(avg_score))
    maddpg.save_checkpoint()
    '''
    return_array = np.array(return_list)
    for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
        plt.figure()
        plt.plot(
            np.arange(return_array.shape[0]) * 100,
            rl_utils.moving_average(return_array[:, i], 9))
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title(f"{agent_name} by MADDPG")'''