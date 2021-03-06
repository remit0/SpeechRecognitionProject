import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
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

    def __init__(self, num_features =512, num_layers = 2):
        super(Network, self).__init__()
        self.gru = nn.GRU(321, hidden_size = num_features, num_layers = num_layers, bidirectional = True, batch_first = True)
        self.fc = nn.Linear(num_features*2, 12)

    def forward(self, x):
        with torch.no_grad():
            inx = torch.ones(x.size(0), 321, 49)
            for i in range(x.size(0)):
                inx[i, :, :] = compute_spec(x[i])
        inx = inx.to(DEVICE)
        inx = torch.transpose(inx,1,2)
        inx, _ = self.gru(inx)
        inx = self.fc(inx[:, -1, :])
        return inx

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