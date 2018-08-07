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

data_path = '/vol/gpudata/rar2417/Data'

output_path = '/vol/gpudata/rar2417/results/model1'

model_path_noise = '/vol/gpudata/rar2417/results/model1/models/onego_108.ckpt'
model_path_no_noise = '/vol/gpudata/rar2417/results/model2/models/onego_64.ckpt'

sys.path.insert(0, source1)

os.chdir(source1)
from dataset_rsb import SRCdataset as data_rsb
from model_rsb import Network as network_rsb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_rsb = data_rsb(data_path + '/submission_list.txt', '/vol/paramonos/datasets/TF-speech/test/test/audio', "submission")

model_no_noise = network_rsb(512, 2).to(device)
model_noise = network_rsb(512, 2).to(device)

model_no_noise.load_state_dict(torch.load(model_path_no_noise))
model_noise.load_state_dict(torch.load(model_path_noise))

model_no_noise.eval()
model_noise.eval()

with open(output_path + '/submission'+KEY+'.csv', 'w', newline='') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['fname', 'label'])

with torch.no_grad():

    for i in range(dataset_rsb.__len__()):
        outputs_no_noise = softmax(model_no_noise(dataset_rsb[i]['audio'].unsqueeze(0).unsqueeze(0).to(device)).squeeze(0), dim = 0)
        outputs_noise = softmax(model_noise(dataset_rsb[i]['audio'].unsqueeze(0).unsqueeze(0).to(device)).squeeze(0), dim = 0)

        result = (outputs_no_noise + outputs_noise) / 2
        _, predicted = torch.max(result.data, 0)

        with open(output_path + '/submission'+KEY+'.csv', 'a') as submission_file:
            writer = csv.writer(submission_file, delimiter=',')
            writer.writerow([dataset_rsb[i]['label'], labels[int(predicted.item())]])