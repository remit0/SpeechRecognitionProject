import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# Home made 
from dataset import SRCdataset
from model import Network, BasicBlock

# pylint: disable=E1101, W0612
"""
La liste des courses : 
"""
"""
sending work to GPU + access server data
verify silence class
what about creating a folder with silence in the server + training_list ?
testing procedure
validation procedure
save results at each step // validation + test set
litterature review + what to do + plan mFCC combine (noisy conditions) spectogram
validation set : early stopping accuracy decrease? fix number of epochs 20
loss and accuracy curves 

just resnet tests
check unknown repartition
"""

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.DoubleTensor')

# Hyperparams
num_epochs = 1
batch_size = 2
learning_rate = 0.003

"""
THREE STEP TRAINING
"""

def training_first_step():
    model = Network(BasicBlock).to(device)

    # Fine tuning - ResNet
    for name, param in model.named_parameters():
        if 'gru' in name:
            param.requires_grad = False
        if 'fc2' in name:
            param.requires_grad = False

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Dataset
    dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
    dataset.reduceDataset(20)
    num_batches = dataset.__len__() // batch_size

    for epoch in range(num_epochs):
        dataset.shuffleUnknown()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        for i_batch, batch in enumerate(dataloader):
            # Forward
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            loss = criterion(outputs, batch['label'])

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Display
            print ('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, i_batch+1, num_batches, loss.item()))
            
            # Save loss
            with open('../Data/results/monitoring/loss_step_1.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')
            
        # Save model, accuracy at each epoch
        torch.save(model.state_dict(),'../Data/results/model_save/model_save_ResNet_'+str(epoch+1)+'.pkl')
        evaluation(model)
    
    data_list, unknown, root_dir = dataset.export()
    return data_list, unknown, root_dir



def training_second_step(data_list, unknown, root_dir):
    model = Network(BasicBlock).to(device)
    model.load_state_dict(torch.load('../Data/results/model_save/model_save_ResNet_'+str(num_epochs)+'.pkl'))

    # Fine tuning - gru & fc2
    for params in model.parameters():
        params.requires_grad = False
    for name, param in model.named_parameters():
        if 'gru' in name:
            param.requires_grad = True
        if 'fc2' in name:
            param.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Dataset
    dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
    dataset.copy(data_list, unknown, root_dir)
    num_batches = dataset.__len__() // batch_size

    for epoch in range(num_epochs):
        dataset.shuffleUnknown()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        for i_batch, batch in enumerate(dataloader):
            # Forward
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            loss = criterion(outputs, batch['label'])

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Display
            print ('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, i_batch+1, num_batches, loss.item()))

            # Save loss
            with open('../Data/results/monitoring/loss_step_2.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')
        
        # Save model, accuracy at each epoch
        torch.save(model.state_dict(),'../Data/results/model_save/model_save_BGRU_'+str(epoch+1)+'.pkl')
        evaluation(model)

    data_list, unknown, root_dir = dataset.export()
    return data_list, unknown, root_dir



def training_third_step(data_list, unknown, root_dir):
    model = Network(BasicBlock).to(device)
    model.load_state_dict(torch.load('../Data/results/model_save/model_save_BGRU_'+str(num_epochs)+'.pkl'))

    # Loss and optimizer
    for params in model.parameters():
        params.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*0.1)

    # Dataset
    dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
    dataset.copy(data_list, unknown, root_dir)
    num_batches = dataset.__len__() // batch_size

    for epoch in range(num_epochs):
        dataset.shuffleUnknown()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        for i_batch, batch in enumerate(dataloader):
            # Forward
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            loss = criterion(outputs, batch['label'])

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Display
            print ('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, i_batch+1, num_batches, loss.item()))
            
            # Save loss
            with open('../Data/results/monitoring/loss_step_3.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')

        # Save model, accuracy at each epoch
        torch.save(model.state_dict(),'../Data/results/model_save/model_save_final_'+str(epoch+1)+'.pkl')
        evaluation(model)



"""
ONE STEP TRAINING
"""
def end_to_end_training():
    model = Network(BasicBlock).to(device)

    # Loss and optimizer ###
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataset
    dataset = SRCdataset('../Data/train/training_list.txt', '../Data/train/audio')
    dataset.reduceDataset(10)
    num_batches = dataset.__len__() // batch_size

    for epoch in range(num_epochs):
        dataset.shuffleUnknown()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for i_batch, batch in enumerate(dataloader):
            # Forward
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            loss = criterion(outputs, batch['label'])

            # Backward and optimize
            #model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Display
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.9f}'
                .format(epoch+1, num_epochs, i_batch+1, num_batches, loss.item()), batch['label'])
            
            # Save loss
            with open('../Data/results/monitoring/Xperience.txt', 'a') as myfile:
                myfile.write(str(loss.item())+'\n')

        # Save model, accuracy at each epoch
        torch.save(model.state_dict(),'../Data/results/model_save/end_to_end.pkl')
        evaluation(model)


"""
ACCURACY TEST
"""
def evaluation(model):
    total, correct = 0, 0
    batchsize = 2
    model.eval()

    # Validation set
    dataset = SRCdataset('../Data/train/validation_list.txt', '../Data/train/audio')
    dataset.reduceDataset(10)
    num_batches = dataset.__len__() // 2
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = True)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'].unsqueeze(1).to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label']).sum().item()
            print('Batch[{}/{}]'.format(i_batch+1, num_batches))

    print('Accuracy of the network : %d %%' % (100 * correct / total))
    with open('../Data/results/monitoring/accuracies.txt', 'a') as f:
        f.write(str(100 * correct / total)+'\n')
    model.train()

#model = Network(BasicBlock).to(device)
#model.load_state_dict(torch.load('../Data/results/model_save/end_to_end.pkl'))
#evaluation(model)

"""
MAIN
"""

# clear previous results
#open('../Data/results/monitoring/accuracies.txt', 'w').close()
#open('../Data/results/monitoring/loss_step_1.txt', 'w').close()
#open('../Data/results/monitoring/loss_step_2.txt', 'w').close()
#open('../Data/results/monitoring/loss_step_3.txt', 'w').close()

#training phase
end_to_end_training()
#data_list, unknown, root_dir = training_first_step()
#training_second_step(data_list, unknown, root_dir)
#training_third_step(data_list, unknown, root_dir)


# pylint: enable=E1101, W0612