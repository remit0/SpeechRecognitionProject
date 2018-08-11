import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
# pylint: disable=E1101, W0612

parser = argparse.ArgumentParser()
parser.add_argument('-key', '--filekey', type = str, help='key for multiple trainings')
parser.add_argument('-lr', '--learning_rate', type = float, help='LEARNING_RATE')
args = parser.parse_args()

source = '/vol/gpudata/rar2417/src/model8'
data_path = '/vol/gpudata/rar2417/Data'
output_path = '/vol/gpudata/rar2417/results/model8'
KEY = ''
if args.filekey is not None:
    KEY = args.filekey

os.chdir(source)
from dataset_top import SRCdataset
from model_top import Network, class_accuracy, accuracy

# Device configuration
start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')
print(device)

# Hyperparams
NUM_EPOCHS = 50
BATCH_SIZE = 20
LEARNING_RATE = 0.0003
if args.learning_rate is not None:
    LEARNING_RATE = args.learning_rate
LAMBDA = 0.87

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
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, LAMBDA)
epoch, estop, maxval, maxind = 0, False, 0, 0

while epoch < NUM_EPOCHS and not estop:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
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
    accuracy(model, device, testset, output_path + '/test_'+KEY+'.txt', 4)

    # Early stopping
    if newval > maxval:
        maxval = newval
        maxind = epoch
        torch.save(model.state_dict(), output_path +'/models/top_'+KEY+'.ckpt')
        class_accuracy(model, device, valset, output_path + '/class_'+KEY+'.txt', batchsize=4)

    if epoch > maxind + 5:
        estop = True
    epoch += 1
    dataset.shuffleUnknown()

print('key  ', KEY)
print('time  ', time.time()-start)
print('epochs  ', epoch)
print('learning_rate  ', LEARNING_RATE)
print('lr_decay  ', LAMBDA)
print('batch_size  ', BATCH_SIZE)