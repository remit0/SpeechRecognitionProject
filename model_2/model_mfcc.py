import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
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

def accuracy(model, device, dataset, filename, batchsize=2):
    total, correct = 0, 0
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = True)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['mfccs'].to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label'].to(device)).sum().item()

    with open(filename, 'a') as f:
        f.write(str(100 * correct / float(total))+'\n')
    model.train()
    return(100 * correct / float(total))