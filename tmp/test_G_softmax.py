import torch
import torch.nn as nn
import torch.nn.functional as F


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