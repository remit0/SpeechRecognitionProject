import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# pylint: disable=E1101, W0612

"""
ResNet architecture adapted to 1-d case from:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.kernel_size = 15
        self.padding = 7
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=self.kernel_size, stride=stride, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)                        
        self.relu = nn.ReLU(inplace=True)                           
        self.conv2 = nn.Conv1d(planes, planes, stride=1, kernel_size=self.kernel_size, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample  
        self.stride = stride                     

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, mode):
        super(ResNet, self).__init__()
        self.mode = mode #added from 3-step training of the model
        self.inplanes = 64
        self.dim = 125 #added for convenience to test different kernel parameters for conv1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=16, padding=38, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, 512)
        
        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(self.dim, 2*self.dim, 5, 2, 0, bias=False),
            nn.BatchNorm1d(2*self.dim),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(2*self.dim, 4*self.dim, 5, 2, 0, bias=False),
            nn.BatchNorm1d(4*self.dim),
            nn.ReLU(True),
            )
        self.backend_conv2 = nn.Sequential(
            nn.Linear(4*self.dim, self.dim),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, 12)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #batch_size x features(512) x sequence_length
        x = torch.transpose(x,1,2)
        x = x.contiguous()
        bs = x.size(0)
        sl = x.size(1)
        x = x.view(bs*sl, -1)
        x = self.fc1(x)

        if self.mode == 1:
            x = x.view(bs, sl, 512) 
            x = self.backend_conv1(x)
            x = torch.mean(x, 2)
            x = self.backend_conv2(x)
        
        else:
            x = x.view(bs, sl, 512)
        
        return x

class GRU(nn.Module):

    def __init__(self, num_features = 512, num_layers = 2):
        super(GRU, self).__init__()
        self.gru = nn.GRU(512, hidden_size = num_features, num_layers = num_layers, bidirectional = True, batch_first = True)
        self.fc2 = nn.Linear(num_features*2, 12)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc2(x[:, -1, :])       
        return x

class Network(nn.Module):
    def __init__(self, num_features = 512, num_layers = 2, mode = 0):
        super(Network, self).__init__()
        self.mode = mode
        self.resnet = ResNet(BasicBlock, mode = mode)
        self.gru = GRU(num_features=num_features, num_layers=num_layers)

    def forward(self, x):
        x = x.type(torch.FloatTensor).unsqueeze(1).to(DEVICE)
        x = self.resnet(x)
        if self.mode != 1:
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