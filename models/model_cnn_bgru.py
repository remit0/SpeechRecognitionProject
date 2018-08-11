import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# pylint: disable=E1101, W0612

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(128)                                                 
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, 512)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.transpose(x, 1, 2)
        x = self.fc(x)

        return x

class GRU(nn.Module):
    # input_size: batch_size x sequence_length x features
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(512, 512, num_layers = 2, bidirectional = True, batch_first = True)
        self.fc2 = nn.Linear(512*2, 12)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc2(x[:, -1, :])
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.cnn = CNN()
        self.gru = GRU()
    
    def forward(self, x):
        x = x.unsqueeze(1).to(DEVICE)
        x = self.cnn(x)
        x = self.gru(x)
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