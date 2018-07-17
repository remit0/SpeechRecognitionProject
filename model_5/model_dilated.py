import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# pylint: disable=E1101, W0612

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)                        
        self.relu = nn.ReLU(inplace=True)                           
        self.conv2 = nn.Conv1d(planes, planes, stride=1, kernel_size=3, padding=1, bias=False)
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
        self.mode = mode
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False) 
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, 512)
        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(500, 2*500, 5, 2, 0, bias=False),
            nn.BatchNorm1d(2*500),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(2*500, 4*500, 5, 2, 0, bias=False),
            nn.BatchNorm1d(4*500),
            nn.ReLU(True),
            )
        self.backend_conv2 = nn.Sequential(
            nn.Linear(4*500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Linear(500, 12)
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
        x = self.layer4(x)              #batchSize x features(512) x seqLen(500)

        x = torch.transpose(x,1,2)
        x = x.contiguous()
        x = x.view(x.size(0)*x.size(1), -1) #batchSize*seqLen x features
        x = self.fc1(x)

        if self.mode == 1:
            x = x.view(-1, 500, 512) 
            x = self.backend_conv1(x)
            x = torch.mean(x, 2)
            x = self.backend_conv2(x)

        return x

class Dilation(nn.Module):

    def __init__(self):
        super(Dilation, self).__init__()
        self.conv = nn.Conv1d(512, 1, 10, dilation=20)
        self.bn = nn.BatchNorm1d(1)                        
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(320, 12)

    def forward(self, x):
        x = x.view(-1, 512, 500)      
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        print(x.size())
        x = self.fc2(x[:, -1, :])
        return x

class Network(nn.Module):
    def __init__(self, mode = 0):
        super(Network, self).__init__()
        self.mode = mode
        self.resnet = ResNet(BasicBlock, mode = mode)
        self.dilation = Dilation()
    
    def forward(self, x):
        x = self.resnet(x)
        if self.mode != 1:
            x = self.dilation(x)
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
            c = (predicted == batch['label']).squeeze()

            for i in range(batchsize):
                label = batch['label'][i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    with open(filename, 'w') as myFile:
        for i in range(12):        
            myFile.write('Accuracy of %5s : %2d %%' % (
            labels[i], 100 * class_correct[i] / class_total[i])+'\n')
    model.train()