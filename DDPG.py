import torch
import torch.nn.functional as F
import numpy as np
from network import TwoLayerFC


def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]],requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i]for i, r in enumerate(torch.rand(logits.shape[0]))])
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)
def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)
def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device, agent_idx, chkpt_dir):
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim, chkpt_dir=chkpt_dir, name=self.agent_name).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,hidden_dim, chkpt_dir=chkpt_dir, name=self.agent_name+'target_actor').to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim, chkpt_dir=chkpt_dir, name=self.agent_name+'_critic').to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1, hidden_dim, chkpt_dir=chkpt_dir, name=self.agent_name+'target_critic').to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.seed = np.random.seed(1)
        self.epsilon = 1.0  # 初始值
        self.epsilon_decay = 0.995  # 每个episode减小的量
        self.epsilon_min = 0.1  # 最小值
    def take_action(self, state, explore=False):
        if np.random.rand() < self.epsilon:
            # 随机选择动作
            action = [0]*12
            if np.random.rand() < 0.9:
                action[np.random.randint(4)] = 1
                return action
            else:
                action[np.random.randint(4, 12)] = 1
                return action
        else:
            action = self.actor(state)
            if explore:
                action = gumbel_softmax(action)
            else:
                action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def step(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)