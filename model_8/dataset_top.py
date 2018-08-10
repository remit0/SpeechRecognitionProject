import torch
import numpy as np
from torch.utils.data import Dataset
from random import randint
from scipy.io.wavfile import read, write
from math import floor
from os import listdir
from os.path import isfile, join
import cv2
from librosa.effects import pitch_shift

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

        if label_idx == 11 and self.train:
            sample = {'audio': self.draw_silence_sample(), 'label': 11}
        else:
            item_path = self.root_dir + '/' + item_name
            _, new_sample = read(item_path)
            if len(new_sample) != seq_length:
                padding = seq_length - len(new_sample)
                new_sample = np.concatenate((new_sample, np.zeros(padding, dtype=int)))
            new_sample = new_sample.astype(np.float)
            self.standardize_audio(new_sample, 20)
            new_sample = new_sample.astype(np.int16)
            if self.train:
                prob = np.random.uniform(0, 1)
                if prob < 0.2:
                    new_sample = self.pitch_shifting(new_sample)
                if prob > 0.2 and prob < 0.4:
                    new_sample = self.speed_tuning(new_sample)
                if prob > 0.4 and prob < 0.6:
                    new_sample = self.time_shifting(new_sample, 4800)
                if prob > 0.6 and prob < 0.8:
                    new_sample = self.add_noise_uniform(new_sample, 0.1)
                
            new_sample = self.filter_banks(new_sample)
            new_sample = torch.from_numpy(new_sample)
            new_sample = new_sample.type(torch.FloatTensor)
            if self.mode != "submission":
                sample = {'audio': new_sample, 'label': label_idx}
            else:
                sample = {'audio': new_sample, 'label': item_name}

        return sample

    def draw_silence_sample(self):
        if self.zero_silence < 185: 
            new_sample = np.zeros(seq_length)
            new_sample = self.filter_banks(new_sample)
            new_sample = torch.from_numpy(new_sample).type(torch.FloatTensor)
            self.zero_silence += 1 
        else:
            selected = self.silence[randint(0, len(self.silence)-1)]
            _, sample = read(self.root_dir+'/_background_noise_/'+selected)
            start_index = randint(0, len(sample)-16000)
            new_sample = sample[start_index:start_index+16000] * np.random.uniform(0, 1)
            new_sample = self.filter_banks(new_sample)
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

    def pitch_shifting(self, sample):
        levels = [-2, -1, 1, 2, None]
        pitch_target = levels[randint(0, len(levels)-1)]
        if pitch_target is None:
            return sample
        else:
            return np.int16(pitch_shift(sample.astype(float), 16000, n_steps = pitch_target))

    def filter_banks(self, sample):
        signal = sample.copy()
        sample_rate = 16000
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_size = 0.025 
        frame_stride = 0.01

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) +\
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # hamming window
        frames *= np.hamming(frame_length)
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        nfilt = 120
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        alpha = np.random.uniform(0.9, 1.1)
        hz_points = np.array([hz_points[i] * alpha if hz_points[i] < (4800 * min(alpha, 1) / alpha) else 8000 - ((8000-4800*min(alpha, 1))/(8000-4800*(min(alpha,1)/alpha)))*(8000-hz_points[i]) for i in range(len(hz_points))])
        
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        return filter_banks # time x features

    def standardize_audio(self, sample, n_chunks):
        chunk_size = int(len(sample) / n_chunks)
        max_volume = 4096 #max(sample)
        min_volume = -4096 #min(sample)
        for i in range(n_chunks):
            audio = sample[i*chunk_size: (i+1)*chunk_size]
            min_audio = min(audio)
            max_audio = max(audio)
            if max_audio - min_audio > 1:
                audio = audio - min_audio * np.ones(chunk_size)
                audio = audio * ((max_volume - min_volume) / (max_audio - min_audio))
                audio = audio + min_volume * np.ones(chunk_size)
                sample[i*chunk_size: (i+1)*chunk_size] = audio
    
        
"""
data_path = '../Data/train'
dataset = SRCdataset(data_path + '/training_list.txt', data_path + '/audio')
x = dataset[75]['audio'].numpy()
dataset.standardize_audio(x, 50)
"""