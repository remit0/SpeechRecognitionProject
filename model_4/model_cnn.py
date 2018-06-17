import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# pylint: disable=E1101, W0612

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, (12, 2), bias = False)
        self.pool = nn.MaxPool2d((6, 2), 2)
        self.conv2 = nn.Conv2d(5, 10, (5, 2), bias = False)
        self.fc1 = nn.Linear(10 * 72 * 11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        # batch_size x 1 x 321 x 49 
        x = self.conv1(x)
        x = F.relu(x)  
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x) # batch_size x 10 x 72 x 11
        x = x.view(-1, 10 * 72 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def accuracy(model, device, dataset, filename, batchsize=2):
    total, correct = 0, 0
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = True)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['spec'].unsqueeze(1).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label'].to(device)).sum().item()

    with open(filename, 'a') as f:
        f.write(str(100 * correct / float(total))+'\n')
    model.train()
    return(100 * correct / float(total))
