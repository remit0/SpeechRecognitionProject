import time
import torch
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
parser.add_argument('-key', '--filekey', type = str, help='key for multiple trainings')
parser.add_argument('-mdl', '--model', type = str, help='path to training save')
parser.add_argument('-e', '--epoch', type = int, help='NUM_EPOCHS')
parser.add_argument('-b', '--batch_size', type = int, help='BATCH_SIZE')
parser.add_argument('-lr', '--learning_rate', type = float, help='LEARNING_RATE')
parser.add_argument('-ft', '--features', type = int, help='NUM_FEATURES')
parser.add_argument('-nl', '--layers', type = int, help='NUM_LAYERS')
parser.add_argument('-md', '--mode', type = int, help='1, 2 or 3')
parser.add_argument('-ld', '--lamb', type = int, help='decay')
args = parser.parse_args()

source = '/vol/gpudata/rar2417/src/model1'
if args.source_path is not None:
    source = args.source_path
data_path = '/vol/gpudata/rar2417/Data'
if args.data_path is not None:
    data_path = args.data_path
output_path = '/vol/gpudata/rar2417/results/model1'
if args.output_path is not None:
    output_path = args.output_path
MODEL = output_path + '/models/model_save_ResNet_1.ckpt'
if args.model is not None:
    MODEL = args.model
KEY = ''
if args.filekey is not None:
    KEY = args.filekey

os.chdir(source)
from dataset_rsb import SRCdataset
from model_rsb import Network, ResNet, BasicBlock, accuracy

# Device configuration
start = time.time()
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
NUM_FEATURES = 512
if args.features is not None:
    NUM_FEATURES = args.features
NUM_LAYERS = 2
if args.layers is not None:
    NUM_LAYERS = args.layers
MODE = 1
if args.mode is not None:
    MODE = args.mode
LAMBDA = 0.87
if args.lamb is not None:
    LAMBDA = args.lamb

# Model & Dataset
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
valset = SRCdataset(data_path + '/validation_list.txt', data_path + '/audio')
dataset.display()

if MODE == 1:
    model = ResNet(BasicBlock, MODE).to(device)
if MODE == 2:
    model = Network(num_features=NUM_FEATURES, num_layers=NUM_LAYERS).to(device)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(MODEL)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for name, param in model.named_parameters():
        if 'gru' in name:
            param.requires_grad = True
        if 'fc2' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
if MODE == 3:
    model = Network(num_features=NUM_FEATURES, num_layers=NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL))
    for params in model.parameters():
        params.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, LAMBDA)
epoch, estop, maxval, maxind = 0, False, 0, 0
num_batches = dataset.__len__() // BATCH_SIZE


while epoch < NUM_EPOCHS and not estop:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    if epoch > 4:
        scheduler.step()
    
    for i_batch, batch in enumerate(dataloader):
        # Forward
        optimizer.zero_grad()
        outputs = model(batch['audio'].unsqueeze(1).to(device))
        loss = criterion(outputs, batch['label'].to(device))

        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Save loss
        with open(output_path +'/loss_'+KEY+'.txt', 'a') as myfile:
            myfile.write(str(loss.item())+'\n')

    # Save model, accuracy at each epoch
    newval = accuracy(model, device, valset, output_path + '/val_'+KEY+'.txt', 4)
    accuracy(model, device, dataset, output_path + '/train_'+KEY+'.txt', 4)
    
    # Early stopping
    if newval > maxval:
        maxval = newval
        maxind = epoch
        if MODE == 1:
            torch.save(model.state_dict(), output_path + '/models/ResNet_'+KEY+'.ckpt')
        if MODE == 2:
            torch.save(model.state_dict(), output_path +'/models/BGRU_'+KEY+'.ckpt')
        if MODE == 3:
            torch.save(model.state_dict(), output_path +'/models/final_'+KEY+'.ckpt')

    if epoch > maxind + 4:
        estop = True
    epoch += 1
    dataset.shuffleUnknown()
    dataset.generateSilenceClass()
