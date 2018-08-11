import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
import sys
# pylint: disable=E1101, W0612
# Remember about output_path / model file

"""
# GPU CLUSTER
source = '/vol/gpudata/rar2417/src/model1'  #path to code location
data_path = '/vol/gpudata/rar2417/Data' #path to the parent directory of 'audio'
output_path = '/vol/gpudata/rar2417/results/model1' #path to output the results
model_path = output_path + '/models/resnet_bgru_1.ckpt' #path to find pre-trained model
"""

# HOME SETUP
source = '/home/r2d9/Desktop/SpeechRecognitionProject/refactored'  #path to code location
data_path = '/home/r2d9/Desktop/Data/train' #path to the parent directory of 'audio'
output_path = '/home/r2d9/Desktop' #path to output the results
model_path = output_path + '/models/model_1.ckpt' #path to find pre-trained model

sys.path.insert(0, source)
sys.path.insert(0, source+'/models')

parser = argparse.ArgumentParser()
parser.add_argument('-key', '--filekey', type = str, help='key for multiple trainings')
parser.add_argument('-lr', '--learning_rate', type = float, help='LEARNING_RATE')
args = parser.parse_args()

KEY = '' #provided for convenience, easy way to differenciate experiments
if args.filekey is not None:
    KEY = args.filekey

from dataset import dataset
#from model_resnet_bgru import Network, accuracy
#from model_mfcc_bgru import Network, accuracy
#from model_spec_bgru import Network, accuracy
#from model_mfrn_bgru import Network, accuracy
#from model_cnn_bgru import Network, accuracy
#from model_fbanks_cnn import Network, accuracy
#from model_resnet_dconv import Network, accuracy
from model_spec_cnn import Network, accuracy

# Configuration
start = time.time()
torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparams
NUM_EPOCHS = 50
BATCH_SIZE = 20
LAMBDA = 0.87
LEARNING_RATE = 0.0003
if args.learning_rate is not None:
    LEARNING_RATE = args.learning_rate

# Model & Dataset
data = dataset(data_path + '/training_list.txt', data_path + '/audio')
valset = dataset(data_path + '/validation_list.txt', data_path + '/audio')
testset = dataset(data_path + '/testing_list.txt', data_path + '/audio')
model = Network().to(device)
for params in model.parameters():
    params.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, LAMBDA)    #learning rate decay, halved every 5 epochs
epoch, estop, maxval, maxind = 0, False, 0, 0

while epoch < NUM_EPOCHS and not estop: #early stopping
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    if epoch > 4:   #fixed learning rate for first 5 epochs
        scheduler.step()
    
    for i_batch, batch in enumerate(dataloader):
        # Forward
        optimizer.zero_grad()
        outputs = model(batch['audio'])
        loss = criterion(outputs, batch['label'].to(device))
        print('bite')
        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Save loss
        with open(output_path +'/loss_'+KEY+'.txt', 'a') as myfile:
            myfile.write(str(loss.item())+'\n')

    # Save model, accuracy at each epoch
    newval = accuracy(model, valset, output_path + '/val_'+KEY+'.txt', 4) #accuracy on validation set for early-stopping
    accuracy(model, dataset, output_path + '/train_'+KEY+'.txt', 4) #accuracy on training set to monitor overfitting
    accuracy(model, testset, output_path + '/test_'+KEY+'.txt', 4) #accuracy on testing set

    # Early stopping
    if newval > maxval:
        maxval = newval
        maxind = epoch
        torch.save(model.state_dict(), output_path + '/models/model_'+KEY+'.ckpt')

    if epoch > maxind + 4:
        estop = True
    epoch += 1
    data.resample_unknown_class()

print('key  ', KEY)
print('time  ', time.time()-start)
print('epochs  ', epoch)