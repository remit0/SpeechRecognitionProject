import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
# pylint: disable=E1101, W0612
parser = argparse.ArgumentParser()
parser.add_argument('-source_path', '--source_path', type = str, help='path to python files')
parser.add_argument('-dp', '--data_path',type = str, help='path to train folder')
parser.add_argument('-op', '--output_path',type = str, help='path to results folder, contains subfolder "models"')
parser.add_argument('-e', '--epoch', type = int, help='NUM_EPOCHS')
parser.add_argument('-b', '--batch_size', type = int, help='BATCH_SIZE')
parser.add_argument('-lr', '--learning_rate', type = float, help='LEARNING_RATE')
parser.add_argument('-key', '--keyName', type = str, help='unique key')
parser.add_argument('-ld', '--lambd', type = float, help='lr decay')
args = parser.parse_args()
start = time.time()

source = '/vol/gpudata/rar2417/src/model4'
if args.source_path is not None:
    source = args.source_path
data_path = '/vol/gpudata/rar2417/Data'
if args.data_path is not None:
    data_path = args.data_path
output_path = '/vol/gpudata/rar2417/results/model4'
if args.output_path is not None:
    output_path = args.output_path

os.chdir(source)
from dataset_cnn import SRCdataset
from model_cnn import Network, accuracy, class_accuracy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Hyperparams
NUM_EPOCHS = 50
if args.epoch is not None:
    NUM_EPOCHS = args.epoch
BATCH_SIZE = 20
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
LEARNING_RATE = 0.003
if args.learning_rate is not None:
    LEARNING_RATE = args.learning_rate
KEY = ''
if args.keyName is not None:
    KEY = args.keyName
LAMBDA = 0.87
if args.lambd is not None:
    LAMBDA = args.lambd

# Model & Dataset
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
valset = SRCdataset(data_path + '/validation_list.txt', data_path + '/audio')
testset = SRCdataset(data_path + '/testing_list.txt', data_path + '/audio')
dataset.display()

model = Network().to(device)
for params in model.parameters():
    params.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #SGD ?
scheduler = ExponentialLR(optimizer, LAMBDA)
epoch, estop, maxval, maxind = 0, False, 0, 0

while epoch < NUM_EPOCHS and not estop:
    if epoch > 4:
        scheduler.step()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    for i_batch, batch in enumerate(dataloader):
        # Forward
        optimizer.zero_grad()
        outputs = model(batch['spec'].unsqueeze(1).to(device))
        loss = criterion(outputs, batch['label'].to(device))

        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Save loss
        with open( output_path + '/loss_'+KEY+'.txt', 'a') as myfile:
            myfile.write(str(loss.item())+'\n')

    # Save model, accuracy at each epoch
    newval = accuracy(model, device, valset, output_path + '/val_'+KEY+'.txt', 4)
    accuracy(model, device, dataset, output_path + '/train_'+KEY+'.txt', 4)
    accuracy(model, device, testset, output_path + '/test_'+KEY+'.txt', 4)

    # Early stopping
    if newval > maxval:
        maxval = newval
        maxind = epoch
        torch.save(model.state_dict(), output_path+'/models/spec_'+KEY+'.ckpt')
        class_accuracy(model, device, valset, output_path + '/class_'+KEY+'.txt', batchsize=4)
    if epoch > maxind + 4:
        estop = True
    dataset.shuffleUnknown()
    epoch += 1

print('key  ', KEY)
print('time  ', time.time()-start)
print('epochs  ', epoch)
print('learning_rate  ', LEARNING_RATE)
print('lr_decay  ', LAMBDA)
print('batch_size  ', BATCH_SIZE)

