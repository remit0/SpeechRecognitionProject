import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from dataset import SRCdataset
from model import Network
from librosa.feature import mfcc
from librosa.display import specshow
# pylint: disable=E1101, W0612

"""
extract MFCCs, derivatives, train BGRU // other things stay the same 40ms window 20s hop size 
"""

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Hyperparams
NUM_EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.003
NUM_FEATURES = 256
NUM_LAYERS = 2

# Model & Dataset
model = Network(num_features=NUM_FEATURES, num_layers=NUM_LAYERS)
dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
valset = SRCdataset('../Data/train/validation_list.txt', '../Data/train/audio')

for params in model.parameters():
    params.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
epoch, estop, maxval, maxind = 0, False, 0, 0
num_batches = dataset.__len__() // BATCH_SIZE

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
        
        # Display
        print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
            .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))
        
        # Save loss
        with open('../Data/results/monitoring/lossie.txt', 'a') as myfile:
            myfile.write(str(loss.item())+'\n')

    # Save model, accuracy at each epoch
    newval = evaluation(model, valset, '../Data/results/monitoring/accuracies1.txt', 4)
    evaluation(model, dataset, '../Data/results/monitoring/accuracies2.txt', 4)
    
    # Early stopping
    if newval > maxval:
        maxval = newval
        maxind = epoch
        torch.save(model.state_dict(), '../Data/results/mfcc.ckpt')
    if epoch > maxind + 4:
        estop = True

    dataset.shuffleUnknown()
    dataset.generateSilenceClass()


"""
ACCURACY
"""

def evaluation(model, dataset, filename, batchsize=2):
    total, correct = 0, 0
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = True)
    num_batches = dataset.__len__() // batchsize
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['mfccs'].to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label'].to(device)).sum().item()
            print('Batch[{}/{}]'.format(i_batch+1, num_batches))
    print('Accuracy of the network : %d %%' % (100 * correct / total))
    with open(filename, 'a') as f:
        f.write(str(100 * correct / float(total))+'\n')
    model.train()
    return(100 * correct / float(total))