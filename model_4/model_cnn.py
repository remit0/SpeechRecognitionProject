import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# pylint: disable=E1101, W0612

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 8, (5, 2), bias = False)
        self.conv2 = nn.Conv2d(8, 8, (5, 2), bias = False)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(8, 16, (7, 3), bias=False)
        self.conv4 = nn.Conv2d(16, 16, (7, 3), bias=False)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(10368, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 12)

    def forward(self, x):
        # batch_size x 1 x 321 x 49
        print(x.size())
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
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
