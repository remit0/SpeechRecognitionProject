import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# Home made 
from dataset import SRCdataset
from model import Network, BasicBlock

# pylint: disable=E1101, W0612
"""
La liste des courses : 
"""
"""
check with pingchuan model / loss curve / learning rate decay divide by half every 5 epochs

extract MFCCs, derivatives, train BGRU // other things stay the same 40ms window 20s hop size python_speech_features librosa 
Log spectrograms features
time series features

Spectrogram + CNN as image

256 features bgru // see for learning rate // layers of bgru 
"""

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False 
torch.set_default_tensor_type('torch.DoubleTensor')

# Hyperparams
NUM_EPOCHS = 2
BATCH_SIZE = 2
LEARNING_RATE = 0.003
NUM_FEATURES = 512
NUM_LAYERS = 2

"""
THREE STEP TRAINING
"""

def training_first_step(dataset, validationset):
    open('../Data/results/monitoring/loss_step_1.txt', 'w').close()
    model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS).to(device)

    # Fine tuning - ResNet
    for name, param in model.named_parameters():
        if 'gru' in name:
            param.requires_grad = False
        #if 'fc2' in name:
        #    param.requires_grad = False

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    num_batches = dataset.__len__() // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        for i_batch, batch in enumerate(dataloader):
            # Forward
            optimizer.zero_grad()
            print(batch['audio'].unsqueeze(1).to(device).size())
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            loss = criterion(outputs, batch['label'].to(device))

            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Display
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
                .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))
            
            # Save loss
            with open('../Data/results/monitoring/loss_step_1.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')
    
        # Save model, accuracy at each epoch
        evaluation(model, validationset, '../Data/results/monitoring/accuracies_val.txt', 4)
        evaluation(model, dataset, '../Data/results/monitoring/accuracies_train.txt', 4)

        dataset.shuffleUnknown()
        dataset.generateSilenceClass()

        torch.save(model.state_dict(),'../Data/results/model_save/model_save_ResNet_'+str(epoch+1)+'.ckpt')



def training_second_step(dataset, validationset, modelsave):
    open('../Data/results/monitoring/loss_step_2.txt', 'w').close()
    model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(modelsave))

    # Fine tuning - gru & fc2
    for params in model.parameters():
        params.requires_grad = False
    for name, param in model.named_parameters():
        if 'gru' in name:
            param.requires_grad = True
        if 'fc2' in name:
            param.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    num_batches = dataset.__len__() // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        for i_batch, batch in enumerate(dataloader):
            # Forward
            optimizer.zero_grad()
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            loss = criterion(outputs, batch['label'].to(device))

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Display
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
                .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))

            # Save loss
            with open('../Data/results/monitoring/loss_step_2.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')
        
        # Save model, accuracy at each epoch
        evaluation(model, validationset, '../Data/results/monitoring/accuracies_val.txt', 4)
        evaluation(model, dataset, '../Data/results/monitoring/accuracies_train.txt', 4)

        dataset.shuffleUnknown()
        dataset.generateSilenceClass()

        torch.save(model.state_dict(),'../Data/results/model_save/model_save_BGRU_'+str(epoch+1)+'.ckpt')



def training_third_step(dataset, validationset, modelsave):
    open('../Data/results/monitoring/loss_step_3.txt', 'w').close()
    model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(modelsave))

    # Loss and optimizer
    for params in model.parameters():
        params.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    num_batches = dataset.__len__() // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        for i_batch, batch in enumerate(dataloader):
            # Forward
            optimizer.zero_grad()
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            loss = criterion(outputs, batch['label'].to(device))

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Display
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
                .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))
            
            # Save loss
            with open('../Data/results/monitoring/loss_step_3.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')
        
        # Save model, accuracy at each epoch
        evaluation(model, validationset, '../Data/results/monitoring/accuracies_val.txt', 4)
        evaluation(model, dataset, '../Data/results/monitoring/accuracies_test.txt', 4)

        dataset.shuffleUnknown()
        dataset.generateSilenceClass()

        torch.save(model.state_dict(),'../Data/results/model_save/model_save_final_'+str(epoch+1)+'.ckpt')


"""
ACCURACY
"""

def evaluation(model, dataset, filename, batchsize=2):
    total, correct = 0, 0
    model = model.eval()

    num_batches = dataset.__len__() // batchsize
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = True)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label'].to(device)).sum().item()
            print('Batch[{}/{}]'.format(i_batch+1, num_batches))

    print('Accuracy of the network : %d %%' % (100 * correct / total))
    with open(filename, 'a') as f:
        f.write(str(100 * correct / total)+'\n')
    model = model.train()

"""
MAIN
"""

# clear previous results
open('../Data/results/monitoring/accuracies_val.txt', 'w').close()
open('../Data/results/monitoring/accuracies_test.txt', 'w').close()

# dataset
dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
dataset.reduceDataset(3)

validationset = SRCdataset('../Data/train/validation_list.txt', '../Data/train/audio')
validationset.reduceDataset(3)

#training phase
training_first_step(dataset, validationset)
#training_second_step(dataset, validationset, '../Data/results/model_save/model_save_ResNet_'+str(NUM_EPOCHS)+'.ckpt')
#training_third_step(dataset, validationset, '../Data/results/model_save/model_save_BGRU_'+str(NUM_EPOCHS)+'.ckpt')
