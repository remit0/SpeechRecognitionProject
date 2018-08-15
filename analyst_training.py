import torch
from torch.nn.functional import softmax
import sys
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
# pylint: disable=E1101, W0612

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key', type = str, help='key')
parser.add_argument('-lr', '--learning_rate', type = float, help='LEARNING_RATE')
args = parser.parse_args()

KEY = ''
if args.key is not None:
    KEY = args.key

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']

#path files and data
source = '/home/r2d9/Desktop/SpeechRecognitionProject'  #path to code location
data_path = '/home/r2d9/Desktop/Data/train' #path to the parent directory of 'audio'
output_path = '/home/r2d9/Desktop' #path to output the results

#load models
#model_path1 = '/vol/gpudata/rar2417/results/model8/models/top_2.ckpt'
#model_path2 = '/vol/gpudata/rar2417/results/model1/models/onego_108.ckpt'
#model_path3 = '/vol/gpudata/rar2417/results/model3/models/spec_3.ckpt'
#model_path4 = '/vol/gpudata/rar2417/results/model4/models/model_302.ckpt'
#model_path5 = '/vol/gpudata/rar2417/results/model2/models/mfcc_8.ckpt'

sys.path.insert(0, source)
sys.path.insert(0, source + '/models')

from dataset import Dataset
from model_fbanks_cnn import Network as network_1
from model_resnet_bgru import Network as network_2
from model_spec_bgru import Network as network_3
from model_spec_cnn import Network as network_4
from model_analyst import Network as analyst, accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Dataset(data_path + '/training_list.txt', data_path + '/audio')
valset = Dataset(data_path + '/validation_list.txt', data_path + '/audio')
data.reduce_dataset(1)
valset.reduce_dataset(1)

model1 = network_1().to(device)
model2 = network_2().to(device)
model3 = network_3().to(device)
model4 = network_4().to(device)
analyzer = analyst().to(device)

for params in model1.parameters():
    params.requires_grad = False
for params in model2.parameters():
    params.requires_grad = False
for params in model3.parameters():
    params.requires_grad = False
for params in model4.parameters():
    params.requires_grad = False

#model1.load_state_dict(torch.load(model_path1))
#model2.load_state_dict(torch.load(model_path2))
#model3.load_state_dict(torch.load(model_path3))
#model4.load_state_dict(torch.load(model_path4))

model1.eval()
model2.eval()
model3.eval()
model4.eval()

LEARNING_RATE = 0.0001
if args.learning_rate is not None:
    LEARNING_RATE = args.learning_rate
NUM_EPOCHS = 10
LAMBDA = 0.87

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, analyzer.parameters()), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, LAMBDA)    #learning rate decay, halved every 5 epochs
epoch, estop, maxval, maxind = 0, False, 0, 0
dataloader = DataLoader(data, batch_size=1, shuffle=True, drop_last=False)

models = [analyzer, model1, model2, model3, model4]

while epoch < NUM_EPOCHS and not estop:

    for i_batch, batch in enumerate(dataloader):
        outputs1 = softmax(model1(batch['audio']).squeeze(0), dim = 0)
        outputs2 = softmax(model2(batch['audio']).squeeze(0), dim = 0)
        outputs3 = softmax(model3(batch['audio']).squeeze(0), dim = 0)
        outputs4 = softmax(model4(batch['audio']).squeeze(0), dim = 0)

        input = torch.cat((outputs1, outputs2, outputs3, outputs4), dim = 0)

        optimizer.zero_grad()
        outputs = analyzer(input.unsqueeze(0))
        loss = criterion(outputs, batch['label'].to(device))
        loss.backward()
        optimizer.step()

        with open(output_path +'/loss_'+KEY+'.txt', 'a') as myfile:
            myfile.write(str(loss.item())+'\n')
        print(loss.item())

    newval = accuracy(models, valset, output_path +'/val_'+KEY+'.txt')
    if newval > maxval:
        maxval = newval
        maxind = epoch
        torch.save(analyzer.state_dict(), output_path + '/models/model_'+KEY+'.ckpt')

    if epoch > maxind:
        estop = True
    
    epoch += 1
    data.resample_unknown_class()


                                               
