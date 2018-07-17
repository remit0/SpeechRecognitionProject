import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from dataset_spec import SRCdataset
from model_spec import Network, accuracy
# pylint: disable=E1101, W0612

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Hyperparams
NUM_EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 0.0003
NUM_FEATURES = 256
NUM_LAYERS = 1
KEY = 'debug'
LAMBDA = 0.87

data_path = '../Data/train'
output_path = '../Data/results'

# Model & Dataset
model = Network(num_features=NUM_FEATURES, num_layers=NUM_LAYERS).to(device)
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
valset = SRCdataset(data_path + '/validation_list.txt', data_path + '/audio')
dataset.reduceDataset(4)
valset.reduceDataset(4)
dataset.display()

for params in model.parameters():
    params.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, LAMBDA)
epoch, estop, maxval, maxind = 0, False, 0, 0
num_batches = dataset.__len__() // BATCH_SIZE

while epoch < NUM_EPOCHS and not estop:
    if epoch > 4:
        scheduler.step()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    for i_batch, batch in enumerate(dataloader):
        # Forward
        optimizer.zero_grad()
        outputs = model(batch['spec'].to(device))
        loss = criterion(outputs, batch['label'].to(device))

        # Backward and optimize
        loss.backward()
        optimizer.step()
                
        # Save loss
        #with open( output_path + '/loss_'+KEY+'.txt', 'a') as myfile:
        #    myfile.write(str(loss.item())+'\n')
        
        # Display
        print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
            .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))

    # Save model, accuracy at each epoch
    newval = accuracy(model, device, valset, output_path + '/test_'+KEY+'.txt', 4)
    accuracy(model, device, dataset, output_path + '/test_'+KEY+'.txt', 4)
    
    # Early stopping
    if newval > maxval:
        maxval = newval
        maxind = epoch
        #torch.save(model.state_dict(), output_path+'/models/spec_'+KEY+'.ckpt')
    if epoch > maxind + 5:
        estop = True
    
    dataset.shuffleUnknown()
    epoch += 1
