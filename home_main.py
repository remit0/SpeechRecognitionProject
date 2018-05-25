import torch
import scipy.io.wavfile as scwav
import numpy as np
import torch.nn as nn
import itertools as itools
from torch.utils.data import Dataset, DataLoader
from math import floor
from random import randint

# pylint: disable=E1101, W0612
"""
Problems to be solved : 
sending work to GPU + access server data
verify silence class
what about creating a folder with silence in the server + training_list ?
testing procedure
validation procedure
save results at each step // validation + test set
litterature review + what to do + plan mFCC combine (noisy conditions) spectogram
randomly select at each epoch 2000 unknown examples
validation set : early stopping accuracy decrease? fix number of epochs 20
loss and accuracy curves 
"""
# Device configuration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.DoubleTensor')

# Hyperparams ###
labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
num_epochs = 1
seq_length = 16000
batch_size = 2
learning_rate = 0.0003

class SRCdataset(Dataset):

    def __init__(self, txt_file, root_dir):
        self.root_dir = root_dir
        #build data set list
        with open(txt_file, 'r') as datalist:
            if 'training' not in txt_file:
                self.data_list = [x.strip() for x in datalist.readlines()]
            else:
                repartition = np.ones((12), dtype = np.int16)
                data = [x.strip() for x in datalist.readlines()]
                unknown = []
                for x in data:
                    xlabel = x.split('/')
                    xlabel = xlabel[0]
                    if xlabel in labels:
                        repartition[labels.index(xlabel)] += 1
                    else:
                        unknown.append(x)
                        data.remove(x)
                for i in range(repartition[0]):
                    sample_index = randint(0, len(unknown)-1)
                    data.append(unknown[sample_index])
                repartition[10] = repartition[0]
                print('Current label distribution :  ', repartition)
                print(len(data), np.sum(repartition))
                self.data_list = data
                


        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        #get name of recording
        item_name = self.data_list[idx]
        #get label of recording according to its name
        label = item_name.split('/')
        label = label[0]
        if label in labels:
            label_idx = labels.index(label)
        else:
            label_idx = 10
        #get path
        item_path = self.root_dir + '/' + item_name
        #read sample from path
        _, new_sample = scwav.read(item_path)
        new_sample = torch.from_numpy(new_sample)
        #padd sample
        if len(new_sample) != seq_length:
            padding = seq_length - len(new_sample)
            new_sample = torch.cat((new_sample, torch.zeros([padding], dtype = torch.short)), 0)
        new_sample = new_sample.type(torch.DoubleTensor).to(device)
        #result
        sample = {'audio': new_sample, 'label': label_idx}
        return sample

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

    def __init__(self, block, layers):
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
    
    def average(self, x):
        avg = torch.zeros(x.size(0), x.size(2))
        for i in range(x.size(0)):
            for j in range(x.size(2)):
                avg[i, j] = torch.sum(x[i, :, j]) / x.size(1)
        return avg

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
        x = x.view(batch_size, -1, 512) #batchSize x seqLen x features
        x, _ = self.gru(x)              #batchSize x seqLen x num_directions * hidden_size

        x = x.contiguous()
        x = x.view(x.size(0)*x.size(1), -1) #batchSize*seqLen x 2*hidden_size
        x = self.fc2(x)                 #batchSize*seqLen x 12
        x = x.view(batch_size, -1, 12)  #batchSize x seqLen x 12
        x = self.softmax(x)             #batchSize x seqLen x 12
        x = self.average(x)             #batchSize x 12

        return x


def training_first_step():
    model = Network(BasicBlock, [2, 2, 2, 2]).to(device)

    # Fine tuning - ResNet ###
    for name, param in model.named_parameters():
        if 'gru' in name:
            param.requires_grad = False
        if 'fc2' in name:
            param.requires_grad = False

    # Loss and optimizer ###
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
    num_batches = dataset.__len__() // batch_size
    for epoch in range(num_epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        for i_batch, batch in enumerate(dataloader):
            # Forward
            outputs = model(batch['audio'].unsqueeze(1))
            loss = criterion(outputs, batch['label'])

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Display
            print ('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                .format(epoch+1, num_epochs, i_batch+1, num_batches, loss.item(), np.exp(loss.item())))
            
            # Save loss
            with open('../Data/loss_step_1.txt', 'a') as myfile:
                myfile.write(loss.item()+'\n')
            
        # Save model, accuracy at each epoch
        torch.save(model.state_dict(),'../Data/model_save_ResNet_'+str(epoch+1)+'.pkl')



def training_second_step():
    model = Network(BasicBlock, [2, 2, 2, 2]).to(device)
    model.load_state_dict(torch.load('../Data/model_save_ResNet_'+str(num_epochs)+'.pkl'))

    for params in model.parameters():
        params.requires_grad = False

    # Fine tuning - gru & fc2 ###
    for name, param in model.named_parameters():
        if 'gru' in name:
            param.requires_grad = True
        if 'fc2' in name:
            param.requires_grad = True

    # Loss and optimizer ###
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
    num_batches = dataset.__len__() // batch_size
    for epoch in range(num_epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        for i_batch, batch in enumerate(dataloader):
            # Forward
            outputs = model(batch['audio'].unsqueeze(1))
            loss = criterion(outputs, batch['label'])

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Display
            print ('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                .format(epoch+1, num_epochs, i_batch+1, num_batches, loss.item(), np.exp(loss.item())))

            # Save loss
            with open('../Data/loss_step_2.txt', 'a') as myfile:
                myfile.write(loss.item()+'\n')
        
        # Save model, accuracy at each epoch
        torch.save(model.state_dict(),'../Data/model_save_BGRU_'+str(epoch+1)+'.pkl')



def training_third_step():
    model = Network(BasicBlock, [2, 2, 2, 2]).to(device)
    model.load_state_dict(torch.load('../Data/model_save_BGRU_'+str(num_epochs)+'.pkl'))

    # Loss and optimizer ###
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*0.1)

    dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
    num_batches = dataset.__len__() // batch_size
    for epoch in range(num_epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        for i_batch, batch in enumerate(dataloader):
            # Forward
            outputs = model(batch['audio'].unsqueeze(1))
            loss = criterion(outputs, batch['label'])

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Display
            print ('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                .format(epoch+1, num_epochs, i_batch+1, num_batches, loss.item(), np.exp(loss.item())))
            
            # Save loss
            with open('../Data/loss_step_3.txt', 'a') as myfile:
                myfile.write(loss.item()+'\n')

        # Save model, accuracy at each epoch
        torch.save(model.state_dict(),'../Data/model_save_final_'+str(epoch+1)+'.pkl')





# Main ###
training_first_step()
training_second_step()
training_third_step()

# pylint: enable=E1101, W0612