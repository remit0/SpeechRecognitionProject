import torch
import numpy as np
import torch.nn as nn
# pylint: disable=E1101, W0612

class Network(nn.Module):

    def __init__(self, num_features = 512, num_layers = 2):
        super(Network, self).__init__()
        self.gru = nn.GRU(69, num_features, num_layers = num_layers, bidirectional = True, batch_first = True)
        self.fc = nn.Linear(num_features*2, 12)

    def forward(self, x):
        x = torch.transpose(x,1,2) #batch_size x sequence_length x features
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x
