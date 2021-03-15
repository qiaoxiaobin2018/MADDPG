import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            # next 1 is origin
            # u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])

            # 产生一个一致的离散随机动作
            good_action = np.zeros(9)
            position_for_one = np.random.randint(0, 9)
            good_action[position_for_one] = 1
            u = good_action

        else:
            # 每个 agent 的 observation 维度不一样
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            # 动作样例 -- [ 14.5000, -19.0049,  -3.1720,   7.9169,   2.9559]
            pi = self.policy.evaluation_actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            # next 2 is origin
            # u += noise
            # u = np.clip(u, -self.args.high_action, self.args.high_action)

        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

