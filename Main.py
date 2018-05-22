import torch
import scipy.io.wavfile as scwav
import numpy as np
import torch.nn as nn
import which_set as ws
#from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable

# pylint: disable=E1101

# Device configuration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.DoubleTensor')

# Hyperparams ###
labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
num_epochs = 5
seq_length = 16000
num_batches = 3
batch_size = 3
learning_rate = 0.0003

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

    def __init__(self, block, layers, num_classes=512):
        self.inplanes = 64
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False) 
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc1 = nn.Linear(512, 512)
        self.gru = nn.GRU(512, 512, num_layers = 2, bidirectional = True, batch_first = True) #hiddensize(512)
        self.fc2 = nn.Linear(512*2, 12) #hiddensize*numdirection
        self.softmax = nn.Softmax(dim=1)

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
        x = self.layer4(x)   #batchSize x features(512) x seqLen(500)

        x = torch.transpose(x,1,2)
        x = x.contiguous()
        x = x.view(x.size(0)*x.size(1), -1) #batchSize*seqLen x features

        x = self.fc1(x)             
        x = x.view(batch_size, -1, 512) #batchSize x seqLen x features
        x, _ = self.gru(x) #batchSize x seqLen x num_directions * hidden_size

        x = x.contiguous()
        x = x.view(x.size(0)*x.size(1), -1) #batchSize*seqLen x 2*hidden_size
        x = self.fc2(x) #batchSize*seqLen x 12
        x = self.softmax(x) #batchSize*seqLen x 12

        return x

model = Network(BasicBlock, [2, 2, 2, 2], 200).to(device)

# Loss and optimizer ###
criterion = nn.CrossEntropyLoss()

#model.gru.parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #see what to upgrade gru frozen

# Read ###
training_list = ws.read_set_file('../Data/train','training')        #how to get data from server ? #pc

# Parameters ###

# Data setup ### What about data <1s ?
#data divide ?
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

#torch.Dataloader('.wav')

data_labels = torch.from_numpy(labels_numpy)
data = torch.from_numpy(data_numpy)
# 2 step training ??
for epoch in range(num_epochs):                 # initial weights ???
    #getitem make batch
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
        optimizer.step()
        
        # Display
        step = i+1 
        #if step % 100 == 0:
        print ('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
            .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

torch.save(model.state_dict(),'model_save.txt')

#model = Network(BasicBlock, [2, 2, 2, 2], 200)
#model.load_state_dict(torch.load('model_save.txt'))

# pylint: enable=E1101