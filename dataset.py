import torch
import numpy as np
from torch.utils.data import Dataset
from random import randint
import scipy.io.wavfile as scwav
# pylint: disable=E1101, W0612

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
seq_length = 16000

class SRCdataset(Dataset):

    def __init__(self, txt_file, root_dir):

        self.root_dir = root_dir
        # Build data set list
        with open(txt_file, 'r') as datalist:
            # For testing and validation we use everthing
            if 'training' not in txt_file:
                self.data_list = [x.strip() for x in datalist.readlines()]
                self.unknown = []
            # For training we have to balance the dataset
            else:
                repartition = np.zeros((12), dtype = np.int16)
                data = [x.strip() for x in datalist.readlines()]
                data_list, unknown_list = [], []
                # Balancing the unknown set
                for x in data:
                    xlabel = x.split('/')
                    xlabel = xlabel[0]
                    if xlabel in labels:
                        repartition[labels.index(xlabel)] += 1
                        data_list.append(x)
                    else:
                        unknown_list.append(x)

                for i in range(repartition[0]):
                    sample_index = randint(0, len(unknown_list)-1)
                    data_list.append(unknown_list[sample_index])
                repartition[10] = repartition[0]
                print('Class distribution :  ', [(labels[i], repartition[i]) for i in range(12)])

                self.data_list = data_list
                self.unknown = unknown_list
                
    def shuffleUnknown(self):
        # Remove previous unknown samples
        new_data_list = []
        ucount = 0
        for x in self.data_list:
            xlabel = x.split('/')
            xlabel = xlabel[0]
            if xlabel in labels:
                new_data_list.append(x)
            else:
                ucount += 1

        # Sample new ones
        for i in range(ucount):
            sample_index = randint(0, len(self.unknown)-1)
            new_data_list.append(self.unknown[sample_index])
        
        self.data_list = new_data_list
    
    def reduceDataset(self, label_size):
        repartition = np.zeros((12), dtype = np.int16)
        new_data_list = []
        for x in self.data_list:
            xlabel = x.split('/')
            xlabel = xlabel[0]
            if xlabel in labels:
                if repartition[labels.index(xlabel)] < label_size:
                    new_data_list.append(x)
                    repartition[labels.index(xlabel)] += 1
                
            else:
                if repartition[10] < label_size:
                    new_data_list.append(x)
                    repartition[10] += 1
        print('Reduced class distribution :  ', [(labels[i], repartition[i]) for i in range(12)])
        self.data_list = new_data_list
    
    def display(self):
        repartition = np.zeros((12), dtype = np.int16)
        for x in self.data_list:
            xlabel = x.split('/')
            xlabel = xlabel[0]
            if xlabel in labels:
                repartition[labels.index(xlabel)] += 1
            else:
                repartition[10] += 1
        print(repartition)
    
    def export(self):
        return self.data_list, self.unknown, self.root_dir

    def copy(self, data_list, unknown, root_dir):
        self.data_list = data_list
        self.unknown = unknown
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Get label of recording
        item_name = self.data_list[idx]
        label = item_name.split('/')
        label = label[0]
        if label in labels:
            label_idx = labels.index(label)
        else:
            label_idx = 10

        # Get sample
        item_path = self.root_dir + '/' + item_name
        _, new_sample = scwav.read(item_path)
        new_sample = torch.from_numpy(new_sample)

        if len(new_sample) != seq_length:
            padding = seq_length - len(new_sample)
            new_sample = torch.cat((new_sample, torch.zeros([padding], dtype = torch.short)), 0)
        new_sample = new_sample.type(torch.DoubleTensor)
        sample = {'audio': new_sample, 'label': label_idx}
        return sample