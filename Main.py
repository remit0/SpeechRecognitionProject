import torch
import scipy.io.wavfile as scwav
import numpy as np
import torch.nn as nn
import which_set as ws
from torch.nn.utils import clip_grad_norm

# pylint: disable=E1101
# Device configuration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet block setup ###
class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  #what do i put here ?
        self.bn1 = nn.BatchNorm1d(planes)                        
        self.relu = nn.ReLU(inplace=True)                           
        self.conv2 = nn.Conv1d(planes, planes, stride=1, kernel_size=3, padding=1, bias=False)   #what do i put here ?
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample            #for links between different blocks
        self.stride = stride                     

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:     #for links between different blocks
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Network : Resnet + BGRU ###
class Network(nn.Module):

    def __init__(self, block, layers, hidden_size, num_classes=1000): #numclasses ? not 1000 but what ?
        self.inplanes = 64
        self.hidden_size = hidden_size
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)       #first convolution -- what there ?
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                                            #blocks actual Network
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                                            #end
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc1 = nn.Linear(512, num_classes)
        self.gru = nn.GRU(num_classes, hidden_size, num_layers = 2, bias = True, 
        bidirectional = True) #not sure if it should be utilised that way
        self.fc2 = nn.Linear(hidden_size, 12, bias=True) #how to do many to many *12
        #self.softmax = nn.Softmax()

        for m in self.modules():            #initialise weights ? to be changed ? initialisation of gru ?
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes: #check for block consistency, what about 0 padding
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
        x, _ = self.gru(x)
        x = self.fc2(x)
        #x = self.softmax(x)

        return x

model = Network(BasicBlock, [2, 2, 2, 2], 150).to(device)

# Backpropagation through time ###
#def detach(states):
#    return [state.detach() for state in states] 

# Loss and optimizer ###
learning_rate = 0.0001              #to be set ? adaptative ?
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Read ###
training_list = ws.read_set_file('../Data/train','training')

# Parameters ###
labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']

# Hyperparams ###
num_epochs = 5
seq_length = 16000
hidden_size = 10
num_layers = 10
num_batches = 200
batch_size = 100
num_train_samples = len(training_list)

# Data setup ###
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

for epoch in range(num_epochs):
    # initial weights ?
    for i in range(0, num_batches):
        # Get mini-batch inputs and targets
        inputs = data[:, i, :].to(device)
        targets = data_labels[:, i, :].to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        # Display
        step = (i+1) // seq_length
        if step % 100 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
            .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

"""
### Train the model ###
for epoch in range(num_epochs):
    
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device), ####
              torch.zeros(num_layers, batch_size, hidden_size).to(device)) ####
    
    # initialisation 
    #

    
    for i in range(0, data.size - seq_length, seq_length):  #data.sze(1)
        # Get mini-batch inputs and targets
        inputs = data[:, i:i+seq_length].to(device)
        targets = data[:, (i+1):(i+1)+seq_length].to(device)
        
        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        if step % 100 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
.format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))
"""



# pylint: enable=E1101