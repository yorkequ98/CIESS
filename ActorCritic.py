import numpy
import numpy as np
import torch
import torch.nn as nn
import Configurations as config


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, init_w=3e-3):
        super(Actor, self).__init__()

        # the input layer and consecutive hidden layers
        self.fc1 = nn.Linear(config.STATE_SIZE, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.init_weights(init_w)  # do not change

    def init_weights(self, init_w):
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        # self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        # the output
        output = self.fc1(x)
        output = self.relu(output)

        output = self.fc2(output)
        output = self.relu(output)

        output = self.fc3(output)
        output = self.tanh(output)

        output = (config.MAX_EMB_SIZE - config.MIN_EMB_SIZE) * (output + 1) / 2 + config.MIN_EMB_SIZE

        return output


class CriticTD(nn.Module):
    def __init__(self):
        super(CriticTD, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(config.STATE_SIZE + config.ACTION_SIZE, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(config.STATE_SIZE + config.ACTION_SIZE, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, xs):
        state, action = xs
        sa = torch.cat([state, action], 1)

        q1 = self.relu(self.l1(sa))
        q1 = self.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = self.relu(self.l4(sa))
        q2 = self.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, xs):
        state, action = xs
        sa = torch.cat([state, action], 1)

        q1 = self.relu(self.l1(sa))
        q1 = self.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

