import torch
import numpy as np
from torch.utils.data import Dataset
from random import randint
from scipy.io.wavfile import write, read
from math import floor
from os import listdir
from os.path import isfile, join
from scipy.signal import spectrogram
# pylint: disable=E1101, W0612

labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
unknown_words = ['bed', 'bird', 'cat', 'dog', 'eight', 'five', 'four', 'happy', 'house', 'marvin', 'nine', 'one', 'seven', 'sheila',
                'six', 'three', 'tree', 'two', 'wow', 'zero']
seq_length = 16000

class SRCdataset(Dataset):

    def __init__(self, txt_file, root_dir):

        self.root_dir = root_dir
        self.txt_file = txt_file

        path = self.root_dir +'/_background_noise_'
        noise_list = [f for f in listdir(path) if isfile(join(path, f))]
        noise_list.remove('README.md')
        self.silence = noise_list

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
                new_sample =  self.draw_silence_sample()
                new_sample = new_sample.astype(float)
            else:
                # get sample
                item_path = self.root_dir + '/' + item_name
                _, new_sample = read(item_path)
                if len(new_sample) != seq_length:
                    padding = seq_length - len(new_sample)
                    new_sample = np.concatenate((new_sample, np.zeros(padding, dtype=int)))
                new_sample = new_sample.astype(float)

            _, _, spectrogram = signal.spectrogram(new_sample, fs=16000, nperseg = 640, noverlap = 320, detrend = False)
            spectrogram = np.log(spectrogram.astype(np.float32) + 1e-10)
            spectrogram = torch.from_numpy(spectrogram)
            spectrogram = spectrogram.type(torch.FloatTensor)
            return {'spec': spectrogram, 'label': label_idx}

        except:
            print("bugged item:", item_name)
            print("label", label_idx, label)
            new_sample = np.zeros(16000)
            new_sample = new_sample.astype(float)
            _, _, spectrogram = signal.spectrogram(new_sample, fs=16000, nperseg = 640, noverlap = 320, detrend = False)
            spectrogram = np.log(spectrogram.astype(np.float32) + 1e-10)
            spectrogram = torch.from_numpy(spectrogram)
            spectrogram = spectrogram.type(torch.FloatTensor)
            return {'spec': spectrogram, 'label': label_idx}