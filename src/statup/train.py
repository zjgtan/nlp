# coding: utf8
import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        fc = F.relu(self.fc1(x))
        return self.fc2(fc)

if __name__ == '__main__':
    net = Network()
    x = [[0, 1, 0, 1, 1, 0, 1, 0, 0, 0]]
    x = torch.FloatTensor(x)
    print(net(x))


