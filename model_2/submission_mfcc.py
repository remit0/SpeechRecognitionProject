import torch
from torch.utils.data import DataLoader
import csv
import argparse
import os
# pylint: disable=E1101, W0612

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']

"""
parser = argparse.ArgumentParser()
parser.add_argument('-mdl', '--model', type = str, help='path to training save')
parser.add_argument('-ft', '--features', type = int, help='NUM_FEATURES')
parser.add_argument('-nl', '--layers', type = int, help='NUM_LAYERS')
args = parser.parse_args()

source = '/vol/gpudata/rar2417/src/model2'
data_path = '/vol/gpudata/rar2417/Data'
output_path = '/vol/gpudata/rar2417/results/model2'
model_path = output_path + '/models/final_21.ckpt'
if args.model is not None:
    model_path = args.model

os.chdir(source)
from dataset_mfcc import SRCdataset
from model_mfcc import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_FEATURES = 512
if args.features is not None:
    NUM_FEATURES = args.features
NUM_LAYERS = 2
if args.layers is not None:
    NUM_LAYERS = args.layers

dataset = SRCdataset(data_path + '/submission_list.txt', '/vol/paramonos/datasets/TF-speech/test/test/audio', "submission")
model = Network(num_features=NUM_FEATURES, num_layers=NUM_LAYERS).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, drop_last = False)

with open(output_path + '/submission.csv', 'w', newline='') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['fname', 'label'])
with torch.no_grad():
    for i_batch, batch in enumerate(dataloader):
        outputs = model(batch['mfccs'].to(device))
        _, predicted = torch.max(outputs.data, 1)
        with open(output_path + '/submission.csv', 'a') as submission_file:
            writer = csv.writer(submission_file, delimiter=',')
            writer.writerow([batch['label'][0], labels[int(predicted.item())]])
"""

from dataset_mfcc import SRCdataset
from model_mfcc import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_FEATURES = 512
NUM_LAYERS = 2

dataset = SRCdataset('../Data/submission_list.txt', '../Data/test/audio', "submission")
dataset.reduceDataset(5)

model = Network(num_features=NUM_FEATURES, num_layers=NUM_LAYERS).to(device)
#model.load_state_dict(torch.load(model_path))
model.eval()

dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, drop_last = False)

with open('submission.csv', 'w', newline='') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['fname', 'label'])
with torch.no_grad():
    for i_batch, batch in enumerate(dataloader):
        outputs = model(batch['mfccs'].to(device))
        _, predicted = torch.max(outputs.data, 1)
        with open('submission.csv', 'a') as submission_file:
            writer = csv.writer(submission_file, delimiter=',')
            writer.writerow([batch['label'][0], labels[int(predicted.item())]])