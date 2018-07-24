import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from dataset_dilated import SRCdataset
from model_dilated import Network, accuracy, class_accuracy
# pylint: disable=E1101, W0612

data_path = '../Data/train'
output_path = '../Data/results'
MODEL = output_path + '/models/ResNet_debug.ckpt'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

# Hyperparams
NUM_EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 0.0003
NUM_FEATURES = 256
NUM_LAYERS = 1
MODE = 4
KEY = 'debug'
LAMBDA = 0.87

# Model & Dataset
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
valset = SRCdataset(data_path + '/validation_list.txt', data_path + '/audio')
dataset.reduceDataset(5)
valset.reduceDataset(1)
dataset.display()

if MODE == 1:
    model = Network(mode=MODE).to(device)
if MODE == 2:
    model = Network().to(device)
    model.load_state_dict(torch.load(MODEL))
    for name, param in model.named_parameters():
        if 'dilation' in name:
            param.requires_grad = True
        if 'resnet' in name:
            param.requires_grad = False
if MODE == 3:
    model = Network().to(device)
    model.load_state_dict(torch.load(MODEL))
    for params in model.parameters():
        params.requires_grad = True
if MODE == 4:
    model = Network().to(device)
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
        
        # Display
        print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
            .format(epoch+1, NUM_EPOCHS, i_batch+1, num_batches, loss.item()))
        
        # Save loss
        #with open(output_path +'/loss_'+KEY+'.txt', 'a') as myfile:
        #    myfile.write(str(loss.item())+'\n')

    # Save model, accuracy at each epoch
    newval = accuracy(model, device, valset, output_path + '/test_'+KEY+'.txt', 4)
    
    # Early stopping
    if newval > maxval:
        maxval = newval
        maxind = epoch
        class_accuracy(model, device, valset, output_path + '/class_'+KEY+'.txt', batchsize=4)
        """
        if MODE == 1:
            torch.save(model.state_dict(), output_path + '/models/ResNet_'+KEY+'.ckpt')
        if MODE == 2:
            torch.save(model.state_dict(), output_path +'/models/BGRU_'+KEY+'.ckpt')
        if MODE == 3:
            torch.save(model.state_dict(), output_path +'/models/final_'+KEY+'.ckpt')
        """
    
    if epoch > maxind + 4:
        estop = True

    epoch += 1
    dataset.shuffleUnknown()

