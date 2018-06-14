import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from dataset_mfcc import SRCdataset
from model_mfcc import Network, accuracy
# pylint: disable=E1101, W0612

"""
extract MFCCs, derivatives, train BGRU // other things stay the same 40ms window 20s hop size 
"""
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Hyperparams
NUM_EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 0.003
NUM_FEATURES = 512
NUM_LAYERS = 2
KEY = 'debug'
STEP_SIZE = 5

data_path = '../Data/train'
output_path = '../Data/results'

# Model & Dataset
model = Network(num_features=NUM_FEATURES, num_layers=NUM_LAYERS).to(device)
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
valset = SRCdataset(data_path + '/validation_list.txt', data_path + '/audio')

print(dataset[0]['mfccs'].size())
"""
for params in model.parameters():
    params.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5)
epoch, estop, maxval, maxind = 0, False, 0, 0

while epoch < NUM_EPOCHS and not estop:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    scheduler.step()
    for i_batch, batch in enumerate(dataloader):
        # Forward
        optimizer.zero_grad()
        outputs = model(batch['mfccs'].to(device))
        loss = criterion(outputs, batch['label'].to(device))

        # Backward and optimize
        loss.backward()
        optimizer.step()
                
        # Save loss
        with open( output_path + '/loss_'+KEY+'.txt', 'a') as myfile:
            myfile.write(str(loss.item())+'\n')

    # Save model, accuracy at each epoch
    newval = accuracy(model, device, valset, output_path + '/accVal_'+KEY+'.txt', 4)
    accuracy(model, device, dataset, output_path + '/accTrain_'+KEY+'.txt', 4)
    
    # Early stopping
    if newval > maxval:
        maxval = newval
        maxind = epoch
        torch.save(model.state_dict(), output_path+'/models/mfcc_'+KEY+'.ckpt')
    if epoch > maxind + 9:
        estop = True
    
    dataset.shuffleUnknown()
    dataset.generateSilenceClass()
    epoch += 1
"""