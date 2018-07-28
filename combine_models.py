import torch
from torch.utils.data import DataLoader
import csv
import os
# pylint: disable=E1101, W0612

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']

"""

source1 = '/vol/gpudata/rar2417/src/model1'
source2 = '/vol/gpudata/rar2417/src/model2'
source3 = '/vol/gpudata/rar2417/src/model3'

data_path = '/vol/gpudata/rar2417/Data'

output_path = '/vol/gpudata/rar2417/results/model5'

model_path1 = output_path + '/models/final_21.ckpt'
model_path2 = output_path + '/models/final_22.ckpt'
model_path3 = output_path + '/models/final_23.ckpt'

os.chdir(source1)
from dataset_rsb import SRCdataset as data_rsb
from model_rsb import Network as network_rsb

os.chdir(source2)
from dataset_mfcc import SRCdataset as data_mfcc
from model_mfcc import Network as network_mfcc

os.chdir(source3)
from dataset_spec import SRCdataset as dataset_spec
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

dataloader1 = DataLoader(dataset_rsb, batch_size = 1, shuffle = False, drop_last = False)
dataloader2 = DataLoader(dataset_mfcc, batch_size = 1, shuffle = False, drop_last = False)
dataloader3 = DataLoader(dataset_spec, batch_size = 1, shuffle = False, drop_last = False)

with open(output_path + '/submission.csv', 'w', newline='') as submission_file:
    writer = csv.writer(submission_file, delimiter=',')
    writer.writerow(['fname', 'label'])

with torch.no_grad():

    for i in range(data_mfcc.__len__()):
        outputs1 = model1(data_rsb[0]['audio'].unsqueeze(1).to(device))
        outputs2 = model2(data_rsb[0]['mfccs'].to(device))
        outputs3 = model2(data_rsb[0]['spec'].to(device))


        #with open(output_path + '/submission.csv', 'a') as submission_file:
        #    writer = csv.writer(submission_file, delimiter=',')
        #    writer.writerow([batch['label'][0], labels[int(predicted.item())]])

"""

source1 = '/home/r2d9/Desktop/SpeechRecognitionProject/model_1'
source2 = '/home/r2d9/Desktop/SpeechRecognitionProject/model_2'
source3 = '/home/r2d9/Desktop/SpeechRecognitionProject/model_3'

data_path = '/home/r2d9/Desktop/Data'

output_path = '/home/r2d9/Desktop'

#model_path1 = output_path + '/models/final_21.ckpt'
#model_path2 = output_path + '/models/final_22.ckpt'
#model_path3 = output_path + '/models/final_23.ckpt'

print(os.getcwd())
#os.chdir(source1)
from dataset_rsb import SRCdataset as data_rsb
from model_rsb import Network as network_rsb

#os.chdir(source2)
from dataset_mfcc import SRCdataset as data_mfcc
from model_mfcc import Network as network_mfcc

#os.chdir(source3)
from dataset_spec import SRCdataset as dataset_spec
from model_spec import Network as network_spec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_rsb = data_rsb(data_path + '/submission_list.txt', '/home/r2d9/Desktop/Data/test/audio', "submission")
dataset_mfcc = data_mfcc(data_path + '/submission_list.txt', '/home/r2d9/Desktop/Data/test/audio', "submission")
dataset_spec = data_spec(data_path + '/submission_list.txt', '/home/r2d9/Desktop/Data/test/audio', "submission")

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

    for i in range(2):
        outputs1 = model1(data_rsb[0]['audio'].unsqueeze(1).to(device))
        outputs2 = model2(data_rsb[0]['mfccs'].to(device))
        outputs3 = model2(data_rsb[0]['spec'].to(device))

        print(outputs1)
        print(" ")
        print(outputs2)
        print(" ")
        print(outputs3)
        print(" ")


        #with open(output_path + '/submission.csv', 'a') as submission_file:
        #    writer = csv.writer(submission_file, delimiter=',')
        #    writer.writerow([batch['label'][0], labels[int(predicted.item())]])
