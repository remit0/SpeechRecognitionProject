import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from dataset_combo import SRCdataset
from model_combo import Network, accuracy, class_accuracy
# pylint: disable=E1101, W0612

data_path = '../Data/train'
output_path = '../Data/results'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Hyperparams
NUM_EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.0003
NUM_FEATURES = 256
NUM_LAYERS = 1
MODE = 4
KEY = 'debug'
LAMBDA = 0.87

# Model & Dataset
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
dataset.reduceDataset(1)
dataset.display()

model = Network(num_features=NUM_FEATURES, num_layers=NUM_LAYERS).to(device)
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
        #outputs = model(batch['audio'].unsqueeze(1).to(device))
        outputs = model(batch['audio'])
        loss = criterion(outputs, batch['label'].to(device))

        # Backward and optimize
        i = 0
        for name, param in model.named_parameters():
            if 'resnet' in name and 'backend' not in name:
                if i == 0:
                    print(name)
                    old = list(param)[0].clone()
                    print(name)
                    i += 1

        loss.backward()
        optimizer.step()

        i = 0
        for name, param in model.named_parameters():
            if 'resnet' in name and 'backend' not in name:
                if i == 0:
                    print(name)
                    new = list(param)[0].clone()
                    print(name)
                    i += 1
        
        print(torch.equal(new.data, old.data))
        
        # Display
        print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
            .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))
        
    epoch += 1
    dataset.shuffleUnknown()

