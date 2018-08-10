import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# pylint: disable=E1101, W0612

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 3), padding = (3, 1))
        self.maxpool1 = nn.MaxPool2d((1, 3))
        self.conv2 = nn.Conv2d(64, 128, (1, 7), padding=(0, 3))
        self.maxpool2 = nn.MaxPool2d((1, 4))
        self.conv3 = nn.Conv2d(128, 256, (1, 10))
        self.conv4 = nn.Conv2d(256, 512, (7, 1), padding=(3, 0))
        self.maxpool3 = nn.MaxPool1d(98) #tbd
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 12)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.squeeze(3)
        x = self.maxpool3(x)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def accuracy(model, device, dataset, filename, batchsize=2):
    total, correct = 0, 0
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = False)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label'].to(device)).sum().item()

    with open(filename, 'a') as f:
        f.write(str(100 * correct / float(total))+'\n')
    model.train()
    return(100*correct/float(total))

def class_accuracy(model, device, dataset, filename, batchsize=2):
    labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = False)
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == batch['label'].to(device)).squeeze()

            for i in range(batchsize):
                label = batch['label'][i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    with open(filename, 'w') as myFile:
        for i in range(12):        
            myFile.write('Accuracy of %5s : %2d %%' % (
            labels[i], 100 * class_correct[i] / class_total[i])+'\n')
    model.train()