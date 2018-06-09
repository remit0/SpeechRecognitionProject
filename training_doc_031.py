import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import argparse
import os
# pylint: disable=E1101, W0612

parser = argparse.ArgumentParser()
parser.add_argument('-source_path', '--source_path', type = str, help='path to python files')
parser.add_argument('-dp', '--data_path',type = str, help='path to train folder')
parser.add_argument('-op', '--output_path',type = str, help='path to results folder, contains subfolder "models"')
parser.add_argument('-mdl', '--model', type = str, help='path to training save')
parser.add_argument('-e', '--epoch', type = int, help='NUM_EPOCHS')
parser.add_argument('-b', '--batch_size', type = int, help='BATCH_SIZE')
parser.add_argument('-lr', '--learning_rate', type = float, help='LEARNING_RATE')
parser.add_argument('-ft', '--features', type = int, help='NUM_FEATURES')
parser.add_argument('-nl', '--layers', type = int, help='NUM_LAYERS')
parser.add_argument('-md', '--mode', type = int, help='1, 2 or 3')
args = parser.parse_args()

source = '/vol/gpudata/rar2417/src'
if args.source_path is not None:
    source = args.source_path
data_path = '/vol/gpudata/rar2417/Data'
if args.data_path is not None:
    data_path = args.data_path
output_path = '/vol/gpudata/rar2417/results'
if args.output_path is not None:
    output_path = args.output_path
MODEL = output_path + '/models/model_save_ResNet_1.ckpt'
if args.model is not None:
    MODEL = args.model

os.chdir(source)

from dataset_031 import SRCdataset
from model_031 import Network, BasicBlock

# Device configuration
use_cuda = torch.cuda.is_available()
print(use_cuda)
torch.backends.cudnn.enabled = True 
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

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
MODE = 1
if args.mode is not None:
    MODE = args.mode

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
        # Tune everything
        for params in model.parameters():
            params.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(NUM_EPOCHS):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        scheduler.step()
        for i_batch, batch in enumerate(dataloader):
            # Forward
            optimizer.zero_grad()
            outputs = model(Variable(batch['audio'].unsqueeze(1).cuda()))
            loss = criterion(outputs, Variable(batch['label'].cuda()))

            # Backward and optimize
            loss.backward()
            optimizer.step()
                        
            # Save loss
            if mode == 1:
                with open(output_path +'/loss_step_1.txt', 'a') as myfile:
                    myfile.write(str(loss.data[0])+'\n')
            if mode == 2:
                with open(output_path +'/loss_step_2.txt', 'a') as myfile:
                    myfile.write(str(loss.data[0])+'\n')
            if mode == 3:
                with open(output_path +'/loss_step_3.txt', 'a') as myfile:
                    myfile.write(str(loss.data[0])+'\n')

        # Save model, accuracy at each epoch
        evaluation(model, validationset, output_path + '/accuracies_val.txt', 4)
        evaluation(model, dataset, output_path + '/accuracies_train.txt', 4)

        dataset.shuffleUnknown()
        dataset.generateSilenceClass()

        if mode == 1:
            torch.save(model.state_dict(), output_path + '/models/model_save_ResNet_'+str(epoch+1)+'.ckpt')
        if mode == 2:
            torch.save(model.state_dict(), output_path +'/models/model_save_BGRU_'+str(epoch+1)+'.ckpt')
        if mode == 3:
            torch.save(model.state_dict(), output_path +'/models/model_save_final_'+str(epoch+1)+'.ckpt')



"""
ACCURACY
"""
def evaluation(model, dataset, filename, batchsize=2):
    total, correct = 0, 0
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = True)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(Variable(batch['audio'].unsqueeze(1).cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == Variable(batch['label'].cuda())).sum().item()

    with open(filename, 'a') as f:
        f.write(str(100 * correct / float(total))+'\n')
    model.train()

"""
MAIN
"""
# dataset
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
validationset = SRCdataset(data_path + '/validation_list.txt', data_path + '/audio')

#training phase
if MODE == 1:
    model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS)
else:
    model = Network(BasicBlock, NUM_FEATURES, NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL))
if use_cuda:
    model.cuda()
training(model, dataset, validationset, MODE)
