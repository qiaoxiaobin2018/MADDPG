
# num_agents = 10
# ll = [5 for i in range(num_agents)]
# print(ll)

# name = 'agent %d' % 121212
# print(name)

# list = [1,2,3,4]

# [agent for agent in self.agents if agent.action_callback is None]

# sub_list = [item for item in list if item>=3]
# print(sub_list)

# import numpy as np
# tt = np.zeros(2)
# print(tt)

import gym
# from gym import spaces
# u_action_space = spaces.Discrete(5)
# print(u_action_space.n)

# action_shape = []
# action_shape.append(5)
# print(action_shape)
#
# x = action_shape[:3]  # 每一维代表该agent的act维度
# print(x)


# ll = [1,2,3,4]
# ll = ll[:4]
# print(ll)

# LL = [[1,2,3]]
# print(LL.sq)

# 测试状态向量
# import numpy as np
# l1 = np.random.uniform(-2, +2, 2)
# l2 = np.random.uniform(-2, +2, 2)
# l3 = np.random.uniform(-2, +2, 2)
# l4 = np.random.uniform(-2, +2, 2)
# l5 = np.random.uniform(-2, +2, 2)
# l6 = np.random.uniform(-2, +2, 2)
#
# print('l1: {}'.format(l1))
# print('l1 - l2: {}'.format(l1 - l2))
#
# entity_pos = []
# for en in range(2):
#     entity_pos.append(l1 - l2)
#
# print('entity_pos: {}'.format(entity_pos))
#
# other_pos = []
# for en in range(3):
#     entity_pos.append(l5 - l2)
#
# print('other_pos: {}'.format(other_pos))
#
# other_vel = []
# res = np.concatenate([l4] + [l4] + entity_pos + other_pos + other_vel)
# print('res: {}'.format(res))


'''
# 测试 G-Softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# define the actor network
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = F.gumbel_softmax(self.action_out(x), tau=1, hard=True)

        return actions

if __name__ == '__main__':
    with torch.no_grad():
        input = torch.rand([14, ])
        net = Actor()
        output = net(input)

        print(output)

        # output_s = output.squeeze(0)
        #
        # print(output_s)
        #
        # u = output_s.cpu().numpy()
        #
        # u = np.clip(u, -1, 1)
        #
        # print(u)
'''


# for i in range(3,4):
#     print(i)

# import numpy as np
# xx  = [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]
# print(xx)

# import numpy as np
# ll = np.array([ 14.5000, -19.0049,  -3.1720,   7.9169,   2.9559])
# print(type(ll))
# lll = [ll]
# print(type(lll))
# print(lll[0][1] - lll[0][2])
# print(lll[0][3] - lll[0][4])

# a = 10
# b = 20

# for i in range(3):
#     print('--{}'.format(i))
#     if i % 100 == 0:
#         ll = [1,2]
#     print('ll:{}'.format(ll[1]))
#     ll = [1,3,2]

# import torch
#
# ob = [[1,2,3],[1,2,3],[1,2,3]]
# uu = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
# state = torch.cat(ob, dim=1)
# print(state)

# import torch
# ll = []
# l1 = torch.tensor([1,2,33,1,1,1,1,121])
# l2 = torch.tensor([1,2,33,1,1,1,1,121])
# l3 = torch.tensor([1,2,33,1,1,1,1,121])
# ll.append(l1)
# ll.append(l2)
# ll.append(l3)
# print(ll)
#
# state = torch.cat(ll, dim=1)


# 测试 critic 网络的输入拼接 -- 横向延展（3个2*5 -- 2*15）
# import numpy as np
# import torch
#
# obs = np.random.randint(-2,2,[2,5])
# obs_t = torch.tensor(obs)
# obs_n = []
# obs_n.append(obs_t)
# obs_n.append(obs_t)
# obs_n.append(obs_t)
# obs_n = obs_n[:3]
#
# print(obs_n)
# state = torch.cat(obs_n, dim=1)
# print(state)


# import numpy as np
#
# x=np.arange(1,2)
# print(x)

# ll = []
# l1 = np.random.randint(-2,2,5)
# print('l1 --',l1)
# l2 = np.random.randint(-2,2,5)
# print('l2 --',l2)
# l1 = torch.tensor(l1)
# print('l1_t --',l1)
# l2 = torch.tensor(l2)
# print('l2_t --',l2)
# ll.append(l1)
# ll.append(l2)
# ll.append(l1)
# print(ll)
# state = torch.cat(ll, dim=1)

# import torch
# import torch.nn.functional as F
#
# logits = torch.tensor([-20,10,9.6,6.2])
# f_softmax = F.softmax(logits, dim=0)
# # Sample soft categorical using reparametrization trick:
# f1  = F.gumbel_softmax(logits, tau=1, hard=False)
# # Sample hard categorical using "Straight-through" trick:
# f2 = F.gumbel_softmax(logits, tau=1, hard=True)
#
# print(logits)
# print('-=-=-==========------------=--')
# print(f_softmax)
# print('-=-=-==========------------=--')
# print(f1)
# print('-=-=-==========------------=--')
# print(f2)
# print('-=-=-==========------------=--')

# import numpy as np
# test = np.zeros(9)
# test[4] = 1
# print(test)
# if test[0]:
#     print('test[0] is True')
# if test[6]:
#     print('test[4] is True')

# rr = np.random.rand()
# xx = rr * 2 - 1
# print('rr: {}  xx: {}'.format(rr, xx))

# print(np.random.randint(0, 10))

import torch

# x = torch.rand([3,])
# print('origin out: ', x)
#
# max_pos = int(torch.max(x, -1)[1])
# x.fill_(0.)
# x[max_pos] = 1.
# print(x)

x = torch.rand([1,9])
print(x)
x = torch.tensor(x, dtype=torch.float32).squeeze(0)
print(x)
# print(x)
# ze  = torch.max(x, -1)[0]
# one = torch.max(x, -1)[1]
# print(ze)
# print(one)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
#
# # define the actor network
# class Actor(nn.Module):
#     def __init__(self):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(14, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 64)
#         self.action_out = nn.Linear(64, 9)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#
#         print(x)
#
#         x =
#
#         return actions
#
# if __name__ == '__main__':
#     with torch.no_grad():
#         input = torch.rand([14, ])
#         net = Actor()
#         output = net(input)
#
#         print(output)

