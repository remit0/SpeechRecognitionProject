import torch
from torch.utils.data import DataLoader
import csv
import os
import argparse
# pylint: disable=E1101, W0612

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']

source = '/vol/gpudata/rar2417/src/model7'
data_path = '/vol/gpudata/rar2417/Data'
output_path = '/vol/gpudata/rar2417/results/model7'
model_path = output_path + '/models/cnnb_1.ckpt'

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key', type = str, help='key')
args = parser.parse_args()

KEY = ''
if args.key is not None:
   KEY = args.key

os.chdir(source)
from dataset_cnnb import SRCdataset
from model_cnnb import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = SRCdataset(data_path + '/submission_list.txt', '/vol/paramonos/datasets/TF-speech/test/test/audio', "submission")
model = Network().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, drop_last = False)

with open(output_path + '/submission'+KEY+'.csv', 'w', newline='') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['fname', 'label'])
with torch.no_grad():
    for i_batch, batch in enumerate(dataloader):
        outputs = model(batch['audio'])
        _, predicted = torch.max(outputs.data, 1)
        with open(output_path + '/submission'+KEY+'.csv', 'a') as submission_file:
            writer = csv.writer(submission_file, delimiter=',')
            writer.writerow([batch['label'][0], labels[int(predicted.item())]])
