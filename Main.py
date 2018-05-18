import torch
import scipy.io.wavfile as scwav
import numpy as np
import torch.nn as nn
import which_set as ws
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable

# pylint: disable=E1101
# Device configuration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.DoubleTensor')

# Hyperparams ###
num_epochs = 5
seq_length = 16000
hidden_size = 10
num_layers = 10
num_batches = 100
batch_size = 100


# ResNet block setup ###
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


# Network : Resnet + BGRU ###
class Network(nn.Module):

    def __init__(self, block, layers, num_classes=1000): #numclasses ?#
        self.inplanes = 64 #ok#
        #self.hidden_size = hidden_size
        super(Network, self).__init__() #ok#
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38,
                               bias=False) #bias? dim of result?(4000)
        self.bn1 = nn.BatchNorm1d(64) #ok#
        self.relu = nn.ReLU(inplace=True) #ok#
        self.maxpool = nn.MaxPool1d(kernel_size=40, stride=2, padding=19) #??? size(2000)#
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #size(250)
        self.avgpool = nn.AvgPool1d(7, stride=1, padding=3) #size(250)
        self.fc1 = nn.Linear((16000/(2**6))*512, num_classes) #numclasses?#
        self.gru = nn.GRU(50, 100, num_layers = 2, bias = True, 
        bidirectional = True, batch_first = True) #hidden_size?# #feature + hiddensize #
        self.fc2 = nn.Linear(num_classes, 12, bias=True) #how to do many to many *12
        #self.softmax = nn.Softmax()
        #What about "silence"?
        #What about weird recordings <1s ?
        for m in self.modules():            #initialise weights ? to be changed ? initialisation of gru ?
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
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(batch_size, -1, 50) #what is features ?
        x, _ = self.gru(x) #what to do with output?
        x = self.fc2(x)
        #x = self.softmax(x)

        return x

model = Network(BasicBlock, [2, 2, 2, 2], 200).to(device)
#print(model)
# Backpropagation through time ###
#def detach(states):
#    return [state.detach() for state in states] 

# Loss and optimizer ###
learning_rate = 0.0003            #to be set ? adaptative ?
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Read ###
training_list = ws.read_set_file('../Data/train','training')

# Parameters ###
labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']

# Data setup ### What about data <1s ?
data_numpy = np.zeros((batch_size, num_batches, seq_length))
labels_numpy = np.zeros((batch_size, num_batches, 12))
it = 0
for i in range(0, num_batches):
    j = 0
    while(j < batch_size):
        _, new_sample = scwav.read(training_list[it][0])
        if len(new_sample) == seq_length:
            data_numpy[j, i, :] = new_sample
            labels_numpy[j, i, :] = ws.label2vector(training_list[it][1])
            j += 1
        it += 1

data_labels = torch.from_numpy(labels_numpy)
data = torch.from_numpy(data_numpy)

for epoch in range(num_epochs):                 # initial weights ???
    for i in range(0, num_batches):
        # Get mini-batch inputs and targets
        inputs = torch.ones([batch_size, 1, seq_length])
        inputs[:, 0, :] = data[:, i, :]
        inputs.to(device)

        targets = torch.ones([batch_size, 12], dtype=torch.long)
        targets[:, :] = data_labels[:, i, :]
        targets.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Display
        step = i+1
        if step % 100 == 0:
            print ('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
            .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))
  
# pylint: enable=E1101