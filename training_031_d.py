import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
# Home made 
from dataset_031 import SRCdataset
from model_031 import Network, BasicBlock

# pylint: disable=E1101, W0612
"""
La liste des courses : 
"""
"""
check with pingchuan learning rate decay divide by half every 5 epochs

extract MFCCs, derivatives, train BGRU // other things stay the same 40ms window 20s hop size python_speech_features librosa 
Log spectrograms features
time series features

Spectrogram + CNN as image

256 features bgru // see for learning rate // layers of bgru 
"""

# Device configuration
use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = True 
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Hyperparams
NUM_EPOCHS = 10
BATCH_SIZE = 5
LEARNING_RATE = 0.003
NUM_FEATURES = 256
NUM_LAYERS = 2

"""
TRAINING
"""

def training(model, dataset, validationset, mode):
    if mode == 1:
        # Fine tuning - ResNet
        for name, param in model.named_parameters():
            if 'gru' in name:
                param.requires_grad = False
            if 'fc2' in name:
                param.requires_grad = False
    
    if mode == 2:
        # Fine tuning - gru & fc2
        for params in model.parameters():
            params.requires_grad = False
        for name, param in model.named_parameters():
            if 'gru' in name:
                param.requires_grad = True
            if 'fc2' in name:
                param.requires_grad = True
    
    if mode == 3:
        for params in model.parameters():
            params.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    num_batches = dataset.__len__() // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        scheduler.step()
        for i_batch, batch in enumerate(dataloader):
            # Forward
            optimizer.zero_grad()
            outputs = model(Variable(batch['audio'].unsqueeze(1).cuda()))
            loss = criterion(Variable(outputs, batch['label'].cuda()))

            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Display
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
                .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))
            
            # Save loss
            if mode == 1:
                with open('../Data/results/monitoring/loss_step_1.txt', 'a') as myfile:
                    myfile.write(str(loss.item())+'\n')
            if mode == 2:
                with open('../Data/results/monitoring/loss_step_2.txt', 'a') as myfile:
                    myfile.write(str(loss.item())+'\n')
            if mode == 3:
                with open('../Data/results/monitoring/loss_step_3.txt', 'a') as myfile:
                    myfile.write(str(loss.item())+'\n')
    
        # Save model, accuracy at each epoch
        evaluation(model, validationset, '../Data/results/monitoring/accuracies_val.txt', 4)
        evaluation(model, dataset, '../Data/results/monitoring/accuracies_train.txt', 4)

        dataset.shuffleUnknown()
        dataset.generateSilenceClass()

        if mode == 1:
            torch.save(model.state_dict(),'../Data/results/model_save/model_save_ResNet_'+str(epoch+1)+'.ckpt')
        if mode == 2:
            torch.save(model.state_dict(),'../Data/results/model_save/model_save_BGRU_'+str(epoch+1)+'.ckpt')
        if mode == 3:
            torch.save(model.state_dict(),'../Data/results/model_save/model_save_final_'+str(epoch+1)+'.ckpt')

"""
ACCURACY
"""

def evaluation(model, dataset, filename, batchsize=2):
    total, correct = 0, 0
    model.eval()

    num_batches = dataset.__len__() // batchsize
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = True)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(Variable(batch['audio'].unsqueeze(1).cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == Variable(batch['label'].cuda())).sum().item()
            print('Batch[{}/{}]'.format(i_batch+1, num_batches))

    print('Accuracy of the network : %d %%' % (100 * correct / total))
    with open(filename, 'a') as f:
        f.write(str(100 * correct / total)+'\n')
    model.train()

"""
MAIN
"""

# dataset
dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')

# validatoinset
validationset = SRCdataset('../Data/train/validation_list.txt', '../Data/train/audio')

# model & training
model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS)
if use_cuda:
    model.cuda()
training(model, dataset, validationset, 1)

#model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS).to(device)
#model.load_state_dict(torch.load( '../Data/results/model_save/model_save_ResNet_'+str(NUM_EPOCHS)+'.ckpt'))
#training(model, dataset, validationset, 2)

#model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS).to(device)
#model.load_state_dict(torch.load('../Data/results/model_save/model_save_BGRU_'+str(NUM_EPOCHS)+'.ckpt'))
#training(model, dataset, validationset, 3)