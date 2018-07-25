import torch
import numpy as np
from torch.utils.data import Dataset
from random import randint
from scipy.io.wavfile import read, write
from math import floor
from os import listdir
from os.path import isfile, join
import cv2

# pylint: disable=E1101, W0612
# 1853
labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
unknown_words = ['bed', 'bird', 'cat', 'dog', 'eight', 'five', 'four', 'happy', 'house', 'marvin', 'nine', 'one', 'seven', 'sheila',
                'six', 'three', 'tree', 'two', 'wow', 'zero']
seq_length = 16000

class SRCdataset(Dataset):

    def __init__(self, txt_file, root_dir, mode = "working"):

        self.root_dir = root_dir
        self.txt_file = txt_file
        self.mode = mode
        self.zero_silence = 0

        if self.mode != "submission":
            path = self.root_dir +'/_background_noise_'
            noise_list = [f for f in listdir(path) if isfile(join(path, f))]
            noise_list.remove('README.md')
            self.silence = noise_list
        else:
            self.silence = []

        with open(txt_file, 'r') as datalist:
            if 'training' not in txt_file:
                self.train = False
                self.data_list = [x.strip() for x in datalist.readlines()]
                self.unknown = []
            else:
                self.train = True
                data = [x.strip() for x in datalist.readlines()]
                data_list, unknown_list = [], []

                for x in data:
                    xlabel = x.split('/')
                    xlabel = xlabel[0]
                    if xlabel in labels:
                        data_list.append(x)
                    else:
                        unknown_list.append(x)

                for i in range(1853):
                    sample_index = randint(0, len(unknown_list)-1)
                    data_list.append(unknown_list[sample_index])
                    data_list.append('silence/silence.wav')

                self.data_list = data_list
                self.unknown = unknown_list

    def shuffleUnknown(self):
        new_data_list, ucount = [], 0
        for x in self.data_list:
            xlabel = x.split('/')
            xlabel = xlabel[0]
            if xlabel in labels:
                new_data_list.append(x)
            else:
                ucount += 1
        for i in range(ucount):
            sample_index = randint(0, len(self.unknown)-1)
            new_data_list.append(self.unknown[sample_index])
        self.data_list = new_data_list

    def reduceDataset(self, label_size):
        repartition, new_data_list = np.zeros((12), dtype = np.int16), []
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
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # get label of recording
        item_name = self.data_list[idx]
        label = item_name.split('/')
        label = label[0]
        if label in labels:
            label_idx = labels.index(label)
        else:
            label_idx = 10
        try:
            if label_idx == 11 and self.train:
                sample = {'audio': self.draw_silence_sample(), 'label': 11}
            else:
                item_path = self.root_dir + '/' + item_name
                _, new_sample = read(item_path)
                if len(new_sample) != seq_length:
                    padding = seq_length - len(new_sample)
                    new_sample = np.concatenate((new_sample, np.zeros(padding, dtype=int)))
                if self.train:
                    new_sample = self.add_noise_kaggle(new_sample)
                    if np.random.uniform(0, 1) < 0.5:
                        new_sample = self.time_shifting(new_sample, 4800)

                new_sample = torch.from_numpy(new_sample)
                new_sample = new_sample.type(torch.FloatTensor)
                if self.mode != "submission":
                    sample = {'audio': new_sample, 'label': label_idx}
                else:
                    sample = {'audio': new_sample, 'label': item_name}
            return sample
        except:
            print("bugged item:", item_name)
            print("label", label_idx, label)
            new_sample = np.zeros(16000)
            new_sample = torch.from_numpy(new_sample)
            new_sample = new_sample.type(torch.FloatTensor)
            return {'audio': new_sample, 'label': 11}
        

    def draw_silence_sample(self):
        if self.zero_silence < 185: 
            new_sample = np.zeros(seq_length)
            new_sample = torch.from_numpy(new_sample).type(torch.FloatTensor)
            self.zero_silence += 1 
        else:
            selected = self.silence[randint(0, len(self.silence)-1)]
            _, sample = read(self.root_dir+'/_background_noise_/'+selected)
            start_index = randint(0, len(sample)-16000)
            new_sample = sample[start_index:start_index+16000] * np.random.uniform(0, 1)
            new_sample = torch.from_numpy(new_sample).type(torch.FloatTensor)

        return new_sample
    
    def add_noise_kaggle(self, sample):
        #randomly draw noise sample
        _, noise =  read(self.root_dir+'/_background_noise_/'+ self.silence[randint(0, len(self.silence)-1)])
        start_index = randint(0, len(noise)-16000)
        noise = noise[start_index:start_index+16000]
        #randomly select noise level
        levels = [-5, 0, 5, 10, None]
        snr_target = levels[randint(0, len(levels)-1)]

        if snr_target is None:
            return sample
        else:
            sample_power = np.sum((sample / 2**15)**2) / len(sample)
            noise_power = np.sum((noise / 2**15)**2) / len(noise)
            factor = np.sqrt((sample_power / noise_power) / (10**(snr_target / 10.0)))
            return np.int16(sample + factor * noise)
    
    def add_noise_uniform(self, sample, factor_max):
        #randomly draw noise sample
        _, noise =  read(self.root_dir+'/_background_noise_/'+ self.silence[randint(0, len(self.silence)-1)])
        start_index = randint(0, len(noise)-16000)
        noise = noise[start_index:start_index+16000]
        return np.int16(sample + np.random.uniform(0, factor_max) * noise)
    
    def time_shifting(self, sample, range):
        shift = randint(-range, range)
        if shift >= 0:
            return np.int16(np.concatenate((sample[shift:], np.random.randint(-32, 32, shift))))
        else:
            return np.int16(np.concatenate((np.random.randint(-32, 32, -shift), sample[:shift])))
    
    def speed_tuning(self, sample):
        speed_rate = np.random.uniform(0.7,1.3)
        sample = sample.astype(float)
        f_sample = cv2.resize(sample, (1, int(len(sample) * speed_rate))).squeeze()
        if len(f_sample) < 16000:
            pad_len = 16000 - len(f_sample)
            f_sample = np.r_[np.random.randint(-32, 32, int(pad_len/2)),
                                f_sample,
                                np.random.randint(-32, 32,int(np.ceil(pad_len/2)))]
            return np.int16(f_sample)
        else:
            cut_len = len(f_sample) - 16000
            f_sample = f_sample[int(cut_len/2):int(cut_len/2)+16000]
            return np.int16(f_sample)


"""
data_path = '../Data/train'
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
x = dataset[75]['audio'].numpy()
print(np.mean(x))
x = dataset.speed_tuning(x)
print(np.mean(x))
"""