import torch
import numpy as np
import torch.nn as nn
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

class Network(nn.Module):

    def __init__(self, block):
        self.inplanes = 64
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False) 
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        self.fc1 = nn.Linear(512, 512)
        self.gru = nn.GRU(512, 512, num_layers = 2, bidirectional = True, batch_first = True) #hiddensize(512)
        self.fc2 = nn.Linear(512*2, 12) #hiddensize*numdirection
        self.softmax = nn.Softmax(dim=2)

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
        bsize = x.size(0)
        x = x.view(x.size(0)*x.size(1), -1) #batchSize*seqLen x features

        x = self.fc1(x)
        x = x.view(bsize, -1, 512) #batchSize x seqLen x features
        x, _ = self.gru(x)              #batchSize x seqLen x num_directions * hidden_size

        x = x.contiguous()
        x = x.view(x.size(0)*x.size(1), -1) #batchSize*seqLen x 2*hidden_size
        x = self.fc2(x)                 #batchSize*seqLen x 12
        x = x.view(bsize, -1, 12)  #batchSize x seqLen x 12
        x = self.softmax(x)             #batchSize x seqLen x 12
        x = torch.mean(x, 1)             #batchSize x 12
 
        return x