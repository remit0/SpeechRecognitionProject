import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import argparse
import os
# pylint: disable=E1101, W0612

parser = argparse.ArgumentParser()
parser.add_argument('source_path', type=str, help='path to python files')
parser.add_argument('data_path', type=str, help='path to train folder')
parser.add_argument('output_path', type=str, help='path to results folder, contains subfolder "models"')
parser.add_argument('model1', type=str, help='path to first step training save')
parser.add_argument('model2', type=str, help='path to second step training save')
parser.add_argument('-e', '--epoch', type = int, help='NUM_EPOCHS')
parser.add_argument('-b', '--batch_size', type = int, help='BATCH_SIZE')
parser.add_argument('-lr', '--learning_rate', type = float, help='LEARNING_RATE')
parser.add_argument('-ft', '--features', type = int , help='NUM_FEATURES')
parser.add_argument('-nl', '--layers', type = int , help='NUM_LAYERS')
args = parser.parse_args()

source = args.source_path
data_path = args.data_path
output_path = args.output_path
MODEL_STEP_1 = args.model1
MODEL_STEP_2 = args.model2

os.chdir(source)

from dataset import SRCdataset
from model import Network, BasicBlock

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False 
torch.set_default_tensor_type('torch.DoubleTensor')

# Hyperparams
NUM_EPOCHS = 20
if args.epoch is not None:
    NUM_EPOCHS = args.epoch
BATCH_SIZE = 20
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
LEARNING_RATE = 0.003
if args.learning_rate is not None:
    LEARNING_RATE = args.learning_rate
NUM_FEATURES = 512
if args.features is not None:
    NUM_FEATURES = args.features
NUM_LAYERS = 2
if args.layers is not None:
    NUM_LAYERS = args.layers


"""
THREE STEP TRAINING
"""

def training_first_step(dataset, validationset):
    open(output_path +'/loss_step_1.txt', 'w').close()
    model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS).to(device)

    # Fine tuning - ResNet
    for name, param in model.named_parameters():
        if 'gru' in name:
            param.requires_grad = False
        if 'fc2' in name:
            param.requires_grad = False

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

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
                        
            # Save loss
            with open(output_path +'/loss_step_1.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')
    
        # Save model, accuracy at each epoch
        evaluation(model, validationset, output_path + '/accuracies_val.txt', 4)
        evaluation(model, dataset, output_path + '/accuracies_train.txt', 4)

        dataset.shuffleUnknown()
        dataset.generateSilenceClass()

        torch.save(model.state_dict(), output_path + '/models/model_save_ResNet_'+str(epoch+1)+'.ckpt')



def training_second_step(dataset, validationset, modelsave):
    open(output_path +'/loss_step_2.txt', 'w').close()
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
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        for i_batch, batch in enumerate(dataloader):
            # Forward
            optimizer.zero_grad()
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            loss = criterion(outputs, batch['label'].to(device))

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Save loss
            with open(output_path +'/loss_step_2.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')
        
        # Save model, accuracy at each epoch
        evaluation(model, validationset, output_path +'/accuracies_val.txt', 4)
        evaluation(model, dataset, output_path +'/accuracies_train.txt', 4)

        dataset.shuffleUnknown()
        dataset.generateSilenceClass()

        torch.save(model.state_dict(), output_path +'/models/model_save_BGRU_'+str(epoch+1)+'.ckpt')



def training_third_step(dataset, validationset, modelsave):
    open(output_path +'/loss_step_3.txt', 'w').close()
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

            # Save loss
            with open(output_path +'/loss_step_3.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')
        
        # Save model, accuracy at each epoch
        evaluation(model, validationset, output_path +'/accuracies_val.txt', 4)
        evaluation(model, dataset, output_path +'/accuracies_train.txt', 4)

        dataset.shuffleUnknown()
        dataset.generateSilenceClass()

        torch.save(model.state_dict(), output_path +'/models/model_save_final_'+str(epoch+1)+'.ckpt')


"""
ACCURACY
"""
def evaluation(model, dataset, filename, batchsize=2):
    total, correct = 0, 0
    model = model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = True)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label'].to(device)).sum().item()

    with open(filename, 'a') as f:
        f.write(str(100 * correct / total)+'\n')
    model = model.train()

"""
MAIN
"""

# clear previous results
#open(output_path + '/accuracies_val.txt', 'w').close()
#open(output_path + '/accuracies_train.txt', 'w').close()

# dataset
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
validationset = SRCdataset(data_path + '/validation_list.txt', data_path + '/audio')

#training phase
training_first_step(dataset, validationset)
#training_second_step(dataset, validationset, MODEL_STEP_1)
#training_third_step(dataset, validatoinset, MODEL_STEP_2)
