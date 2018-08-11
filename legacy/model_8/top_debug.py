import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from dataset_top import SRCdataset
from model_top import Network, accuracy, class_accuracy
# pylint: disable=E1101, W0612

data_path = '../Data/train'
output_path = '../Data/results'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Hyperparams
NUM_EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.00003
KEY = 'debug'
LAMBDA = 0.87

# Model & Dataset
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
dataset.reduceDataset(15)
dataset.display()

model = Network().to(device)
for params in model.parameters():
    params.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, LAMBDA)
epoch, maxval, maxind = 0, 0, 0
num_batches = dataset.__len__() // BATCH_SIZE

while epoch < NUM_EPOCHS:
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
        
        # Display
        print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
            .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))

    epoch += 1