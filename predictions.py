import torch
from torch.nn.functional import softmax
import csv
import sys
import os
import argparse
from torch.utils.data import DataLoader
# pylint: disable=E1101, W0612

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key', type = str, help='key')
args = parser.parse_args()

KEY = ''
if args.key is not None:
    KEY = args.key

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']

#path files and data
source = '/home/r2d9/Desktop/SpeechRecognitionProject'
data_path = '/home/r2d9/Desktop/Data'
output_path = '/home/r2d9/Desktop'

#model_path1 = '/vol/gpudata/rar2417/results/model1/models/final_21.ckpt'
#model_path2 = '/vol/gpudata/rar2417/results/model2/models/final_21.ckpt'
#model_path3 = '/vol/gpudata/rar2417/results/model3/models/final_21.ckpt'

sys.path.insert(0, source)
sys.path.insert(0, '/home/r2d9/Desktop/SpeechRecognitionProject/models')
from dataset import Dataset
from model_resnet_bgru import Network as network_rsb
from model_mfcc_bgru import Network as network_mfcc
#from models import Network as network_spec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Dataset(data_path + '/submission_list.txt', data_path + '/test/audio', "submission")

model1 = network_rsb().to(device)
model2 = network_mfcc().to(device)
#model3 = network_spec().to(device)

#model1.load_state_dict(torch.load(model_path1))
#model2.load_state_dict(torch.load(model_path2))
#model3.load_state_dict(torch.load(model_path3))

model1.eval()
model2.eval()
#model3.eval()

with open(output_path + '/submission'+KEY+'.csv', 'w', newline='') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['fname', 'label'])

with torch.no_grad():
    dataloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=False)
    for i_batch, batch in enumerate(dataloader):
        outputs1 = softmax(model1(batch['audio']).squeeze(0), dim = 0)
        outputs2 = softmax(model2(batch['audio']).squeeze(0), dim = 0)
        #outputs3 = softmax(model3(batch['audio']), dim = 0)

        result = (outputs1+outputs2)/2
        print(result)
        _, predicted = torch.max(result.data, 0)

        with open(output_path + '/submission'+KEY+'.csv', 'a') as submission_file:
            writer = csv.writer(submission_file, delimiter=',')
            writer.writerow([batch['label'][0], labels[int(predicted.item())]])
