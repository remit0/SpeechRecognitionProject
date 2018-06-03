import torch
import numpy as np
from torch.utils.data import Dataset
from random import randint
import scipy.io.wavfile as scwav
from math import floor
from os import listdir
from os.path import isfile, join
from data_setup import clear_silence
# pylint: disable=E1101, W0612

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
unknown_words = ['bed', 'bird', 'cat', 'dog', 'eight', 'five', 'four', 'happy', 'house', 'marvin', 'nine', 'one', 'seven', 'sheila',
                'six', 'three', 'tree', 'two', 'wow', 'zero']
seq_length = 16000

class SRCdataset(Dataset):

    def __init__(self, txt_file, root_dir):

        self.root_dir = root_dir
        self.txt_file = txt_file

        # Build data set list
        with open(txt_file, 'r') as datalist:
            # For testing and validation we use everthing
            if 'training' not in txt_file:
                self.data_list = [x.strip() for x in datalist.readlines()]
                self.unknown = []
            # For training we have to balance the dataset
            else:
                clear_silence(txt_file)
                data = [x.strip() for x in datalist.readlines()]
                data_list, unknown_list = [], []
                # Balancing the unknown set
                for x in data:
                    xlabel = x.split('/')
                    xlabel = xlabel[0]
                    if xlabel in labels:
                        data_list.append(x)
                    else:
                        unknown_list.append(x)

                for i in range(len(data_list)//10):
                    sample_index = randint(0, len(unknown_list)-1)
                    data_list.append(unknown_list[sample_index])

                self.data_list = data_list
                self.unknown = unknown_list
                self.generateSilenceClass()
        self.display()

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

    def unknownStats(self):
        repartition = np.zeros((len(unknown_words)),  dtype = np.int16)
        for x in self.data_list:
            xlabel = x.split('/')
            xlabel = xlabel[0]
            if xlabel in unknown_words:
                repartition[unknown_words.index(xlabel)] += 1
        print('Unknown distribution :  ', [(unknown_words[i], floor(100*repartition[i]/np.sum(repartition))) for i in range(len(repartition))])

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
        print('class distribution :  ', [(labels[i], repartition[i]) for i in range(12)])
    
    def generateSilenceClass(self):
        #clearing silence
        f = open(self.txt_file,'r')
        lines = f.readlines()
        f.close()
        f = open(self.txt_file,'w')
        for line in lines:
            if 'silence' not in line:
                f.write(line)
        f.close()

        self.data_list = [x for x in self.data_list if 'silence' not in x]
        nsamples = len(self.data_list)//11
        path = self.root_dir +'/_background_noise_'
        noise_list = [f for f in listdir(path) if isfile(join(path, f))]
        noise_list.remove('README.md')
        
        for i in range(nsamples):
            #select random noise effect
            selected = noise_list[randint(0, len(noise_list)-1)]
            _, sample = scwav.read('../Data/train/audio/_background_noise_/'+selected)
            #select random start index over 60s
            start_index = randint(0, len(sample)-16000)
            #copy 1s after start index
            new_sample = sample[start_index:start_index+16000]
            new_sample = np.rint(new_sample).astype('int16')
            #write file
            scwav.write('../Data/train/audio/silence/silent'+str(i)+'.wav', 16000, new_sample)
        
        with open(self.txt_file, 'a') as myfile:
            noise_list = [f for f in listdir('../Data/train/audio/silence') if isfile(join('../Data/train/audio/silence', f))]
            for i in range(nsamples):
                myfile.write('silence/'+noise_list[i]+'\n')
                self.data_list.append('silence/'+noise_list[i])

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