import torch
from torch.nn.functional import softmax
import csv
import sys
import os
import argparse
# pylint: disable=E1101, W0612

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key', type = str, help='key')
args = parser.parse_args()

KEY = ''
if args.key is not None:
    KEY = args.key

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']

source1 = '/vol/gpudata/rar2417/src/model1'
source2 = '/vol/gpudata/rar2417/src/model2'
source3 = '/vol/gpudata/rar2417/src/model3'

data_path = '/vol/gpudata/rar2417/Data'

output_path = '/vol/gpudata/rar2417/results/model5'

model_path1 = '/vol/gpudata/rar2417/results/model1/models/final_21.ckpt'
model_path2 = '/vol/gpudata/rar2417/results/model2/models/final_21.ckpt'
model_path3 = '/vol/gpudata/rar2417/results/model3/models/final_21.ckpt'

sys.path.insert(0, source1)
sys.path.insert(0, source2)
sys.path.insert(0, source3)

os.chdir(source1)
from dataset_rsb import SRCdataset as data_rsb
from model_rsb import Network as network_rsb

os.chdir(source2)
from dataset_mfcc import SRCdataset as data_mfcc
from model_mfcc import Network as network_mfcc

os.chdir(source3)
from dataset_spec import SRCdataset as data_spec
from model_spec import Network as network_spec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_rsb = data_rsb(data_path + '/submission_list.txt', '/vol/paramonos/datasets/TF-speech/test/test/audio', "submission")
dataset_mfcc = data_mfcc(data_path + '/submission_list.txt', '/vol/paramonos/datasets/TF-speech/test/test/audio', "submission")
dataset_spec = data_spec(data_path + '/submission_list.txt', '/vol/paramonos/datasets/TF-speech/test/test/audio', "submission")

model1 = network_rsb(512, 2).to(device)
model2 = network_mfcc(512, 2).to(device)
model3 = network_spec(512, 2).to(device)

model1.load_state_dict(torch.load(model_path1))
model2.load_state_dict(torch.load(model_path2))
model3.load_state_dict(torch.load(model_path3))

model1.eval()
model2.eval()
model3.eval()

with open(output_path + '/submission'+KEY+'.csv', 'w', newline='') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['fname', 'label'])

with torch.no_grad():

    for i in range(dataset_rsb.__len__()):
        outputs1 = softmax(model1(dataset_rsb[i]['audio'].unsqueeze(0).unsqueeze(0).to(device)).squeeze(0), dim = 0)
        outputs2 = softmax(model2(dataset_mfcc[i]['mfccs'].unsqueeze(0).to(device)).squeeze(0), dim = 0)
        outputs3 = softmax(model3(dataset_spec[i]['spec'].unsqueeze(0).to(device)).squeeze(0), dim = 0)

        result = (outputs1+outputs2+outputs3)/3
        _, predicted = torch.max(result.data, 0)

        with open(output_path + '/submission'+KEY+'.csv', 'a') as submission_file:
            writer = csv.writer(submission_file, delimiter=',')
            if dataset_rsb[i]['label'] == dataset_mfcc[i]['label'] and dataset_mfcc[i]['label'] == dataset_spec[i]['label']:
                writer.writerow([dataset_rsb[i]['label'], labels[int(predicted.item())]])
            else:
                writer.writerow(['failed', labels[int(predicted.item())]])

"""
source1 = '/home/remito/Desktop/SpeechRecognitionProject/model_1'
source2 = '/home/remito/Desktop/SpeechRecognitionProject/model_2'
source3 = '/home/remito/Desktop/SpeechRecognitionProject/model_3'

data_path = '/home/remito/Desktop/Data'

output_path = '/home/remito/Desktop'

#model_path1 = output_path + '/models/final_21.ckpt'
#model_path2 = output_path + '/models/final_22.ckpt'
#model_path3 = output_path + '/models/final_23.ckpt'

sys.path.insert(0, source1)
sys.path.insert(0, source2)
sys.path.insert(0, source3)

print(sys.path)

os.chdir(source1)
from dataset_rsb import SRCdataset as data_rsb
from model_rsb import Network as network_rsb

os.chdir(source2)
from dataset_mfcc import SRCdataset as data_mfcc
from model_mfcc import Network as network_mfcc

os.chdir(source3)
from dataset_spec import SRCdataset as data_spec
from model_spec import Network as network_spec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_rsb = data_rsb(data_path + '/submission_list.txt', '/home/remito/Desktop/Data/test/audio', "submission")
dataset_mfcc = data_mfcc(data_path + '/submission_list.txt', '/home/remito/Desktop/Data/test/audio', "submission")
dataset_spec = data_spec(data_path + '/submission_list.txt', '/home/remito/Desktop/Data/test/audio', "submission")

model1 = network_rsb(512, 2).to(device)
model2 = network_mfcc(512, 2).to(device)
model3 = network_spec(512, 2).to(device)

#model1.load_state_dict(torch.load(model_path1))
#model2.load_state_dict(torch.load(model_path2))
#model3.load_state_dict(torch.load(model_path3))

model1.eval()
model2.eval()
model3.eval()

with open(output_path + '/submission.csv', 'w', newline='') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['fname', 'label'])

with torch.no_grad():

    for i in range(10):
        outputs1 = softmax(model1(dataset_rsb[i]['audio'].unsqueeze(0).unsqueeze(0).to(device)).squeeze(0), dim = 0)
        outputs2 = softmax(model2(dataset_mfcc[i]['mfccs'].unsqueeze(0).to(device)).squeeze(0), dim = 0)
        outputs3 = softmax(model3(dataset_spec[i]['spec'].unsqueeze(0).to(device)).squeeze(0), dim = 0)

        result = (outputs1+outputs2+outputs3)/3
        _, predicted = torch.max(result.data, 0)

        with open(output_path + '/submission.csv', 'a') as submission_file:
            writer = csv.writer(submission_file, delimiter=',')
            if dataset_rsb[i]['label'] == dataset_mfcc[i]['label'] and dataset_mfcc[i]['label'] == dataset_spec[i]['label']:
                writer.writerow([dataset_rsb[i]['label'], labels[int(predicted.item())]])
            else:
                writer.writerow(['failed', labels[int(predicted.item())]])

"""

