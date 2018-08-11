import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import signal
# pylint: disable=E1101, W0612

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

def compute_spec(sample):
    audio = sample.numpy()
    _, _, spectrogram = signal.spectrogram(audio, fs=16000, nperseg = 640, noverlap = 320, detrend = False)
    spectrogram = np.log(spectrogram.astype(np.float32) + 1e-10)
    spectrogram = torch.from_numpy(spectrogram)
    spectrogram = spectrogram.type(torch.FloatTensor)
    return spectrogram

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

    def forward(self, inp):
        # batch_size x 1 x 321 x 49(time)
        with torch.no_grad():
            x = torch.ones(inp.size(0), 321, 49)
            for i in range(inp.size(0)):
                x[i, :, :] = compute_spec(inp[i])
        
        x = x.unsqueeze(1).to(DEVICE)
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

def accuracy(model, dataset, filename, batchsize=2):
    """
    Computes overall accuracy on the dataset provided
    """
    total, correct = 0, 0
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = False)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'])
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label'].to(DEVICE)).sum().item()

    with open(filename, 'a') as f:
        f.write(str(100 * correct / float(total))+'\n')
    model.train()
    return(100*correct/float(total))

def class_accuracy(model, dataset, filename, batchsize=2):
    """
    Computes per class accuracy on the dataset provided
    """
    labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = False)
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'])
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == batch['label'].to(DEVICE)).squeeze()

            for i in range(batchsize):
                label = batch['label'][i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    with open(filename, 'w') as myFile:
        for i in range(12):        
            myFile.write('Accuracy of %5s : %2d %%' % (
            labels[i], 100 * class_correct[i] / class_total[i])+'\n')
    model.train()