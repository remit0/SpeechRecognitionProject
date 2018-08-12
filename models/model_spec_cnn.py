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
    spectrogram = np.log(spectrogram.astype(np.float32) + 1e-10).T
    spectrogram = torch.from_numpy(spectrogram)
    spectrogram = spectrogram.type(torch.FloatTensor)
    return spectrogram

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 64, (3, 7), padding = (1, 3))
        self.maxpool1 = nn.MaxPool2d((1, 5))
        self.conv2 = nn.Conv2d(64, 128, (1, 7), padding = (0, 3))
        self.maxpool2 = nn.MaxPool2d((1, 5))
        self.conv3 = nn.Conv2d(128, 256, (1, 12))
        self.conv4 = nn.Conv2d(256, 512, (5, 1), padding=(2, 0))
        self.maxpool3 = nn.MaxPool1d(49)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 12)

    def forward(self, inp):
        # batch_size x 1 x 49(time) x 321(frequency)
        with torch.no_grad():
            x = torch.ones(inp.size(0), 49, 321)
            for i in range(inp.size(0)):
                x[i, :, :] = compute_spec(inp[i])
        
        x = x.unsqueeze(1).to(DEVICE)
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