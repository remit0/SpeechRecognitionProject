import torch
from torch.utils.data import Dataset
import numpy as np
from random import randint
from scipy.io.wavfile import read #reading .wav files
from os import listdir #parse directories
from os.path import isfile, join #parse directories
import cv2 #resizing vectors with interpolation of values
from librosa.effects import pitch_shift #audio processing
# pylint: disable=E1101, W0612

LABELS = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence'] #10 commands + silence and unknown words
SEQ_LENGTH = 16000 #sampling rate of audio files is 16kHz, all audio files should be 1s long

class Dataset(Dataset):
    """
    Inherits torch.utils.data.Dataset
    This class aims to make our dataset accessible conveniently so that accessing the k-th item of the dataset
    can be simply done by writing dataset[k]. In addition, PyTorch provides a DataLoader class that works together
    with the Dataset class to easily construct batches (with some nice features such as randomizing batches).
    The dataset can be found at https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data.
    File disposition is supposed to be the same as in the link above.
    """
    """
    This class returns a dict with 'audio' being an np.array of type float of length 16000 and 'label' is the index of
    the target label accordin to the LABELS list.
    """
    """
    During training, I use every sample of the 10 commands as these are 'static' classes, meaning that at each epoch, the
    samples used for these classes are the same. Note: these 10 classes are balanced with around 1850 samples per class.

    The kaggle dataset does not include samples for the silence class but samples of different kinds of noise. Therefore,
    we have to generate the silence class ourselves. This class should not be static as there is not one fixed representation
    of what a silent audio looks like. To keep the dataset balanced, I generate, at each epoch, around 1850 'silence' samples.
    10% of these samples are completely silent (zero vector) and the others are randomly generated (cf. generate_silence_sample)
    as it seems to work well.

    For the unknown class, we dispose of some words that are not commands. There are a lot more samples provided for the unknown
    class than for any other class. To keep the dataset balanced, I choose to use only 1850 samples at each epoch.
    This class is the most tricky because we cannot train the network on any other existing word that is not a command. Our goal
    is to train the network on some words given in the dataset and hope for a good generalization to every other words. Therefore,
    this class is dynamic alike the silence class. I do not keep training on the same samples at each epoch as we do not want our
    network to have a set representation of what 'unknown' is. Instead, I randomly select 1850 samples of the unknown class at each
    epoch which is possible due to the sizable number of samples provided for the unknown class. 
    """

    def __init__(self, txt_file, root_dir, mode = "training"):

        self.txt_file = txt_file #text file listing all the audio composing the dataset (has to be made cf. data_setup.py)
                                 #training, testing and validation list can be made using the function provided by kaggle.
                                 #i used 80% for training, 10% for validation and 10% for testing
        self.root_dir = root_dir #directory where to find the audio files
        self.mode = mode #account for different files organization for the testing set of the competition (as the 'competition set' is unlabelled)
        self.silence_class_zeros_count = 0 #ensure that 10% of the silence class are purely silent samples.
        self.noise_list = [] #list available noise for data augmentation and silence class
        self.unknown_list = [] #list all the unknown samples available
        self.data_list = [] #list all the samples belonging to the dataset we crafted
        self.train = True #randomly crafting classes only occurs while training

        if self.mode != "submission":
            path = self.root_dir +'/_background_noise_'
            noise_list = [f for f in listdir(path) if isfile(join(path, f))]
            noise_list.remove('README.md')
            self.noise_list = noise_list

        with open(txt_file, 'r') as data:
            if 'training' not in txt_file: #when not traning we simply use every sample available
                self.train = False
                self.data_list = [x.strip() for x in data.readlines()]
            else:
                full_data_list = [x.strip() for x in data.readlines()]
                for x in full_data_list:
                    xlabel = x.split('/')
                    xlabel = xlabel[0]
                    if xlabel in LABELS:
                        self.data_list.append(x)
                    else:
                        self.unknown_list.append(x)

                for i in range(1850):
                    sample_index = randint(0, len(self.unknown_list)-1)
                    self.data_list.append(self.unknown_list[sample_index])
                    self.data_list.append('silence/silence.wav') #we do not actually have 'silence' audio as we generate directly arrays
                                                                 #as for every other class we read actual files from path, it is a trick 
                                                                 #to fit the silence class with the other classes.
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item_name = self.data_list[idx]
        label = item_name.split('/')
        label = label[0]
        if label in LABELS:
            label_idx = LABELS.index(label)
        else:
            label_idx = 10
        
        try:
            if label_idx == 11 and self.train:
                sample = {'audio': self.generate_silence_sample(), 'label': 11}
            else:
                item_path = self.root_dir + '/' + item_name
                _, new_sample = read(item_path)
                if len(new_sample) != SEQ_LENGTH: #padding on inputs that are less than 1s
                    padding = SEQ_LENGTH - len(new_sample)
                    new_sample = np.concatenate((new_sample, np.zeros(padding, dtype=int)))
                if self.train:
                    prob = np.random.uniform(0, 1)
                    if prob < 0.2:
                        new_sample = self.pitch_shifting(new_sample)
                    if prob > 0.2 and prob < 0.4:
                        new_sample = self.speed_tuning(new_sample)
                    if prob > 0.4 and prob < 0.6:
                        new_sample = self.time_stretching(new_sample, 4800)
                    if prob > 0.6 and prob < 0.8:
                        new_sample = self.add_noise_uniform(new_sample, 0.1)
                new_sample = new_sample.astype(np.float32)
                if self.mode != "submission": #for training we are interested in target label and file name for submitting on kaggle
                    sample = {'audio': new_sample, 'label': label_idx}
                else:
                    sample = {'audio': new_sample, 'label': item_name}
            return sample
        
        except:
            print("bugged item:", item_name)
            print("label", label_idx, label)
            new_sample = np.zeros(16000, dtype = np.int16)
            return {'audio': new_sample, 'label': 11}
        
    def resample_unknown_class(self):
        """
        When traning, this function is called once per epoch to draw new unknown samples to learn from
        """
        new_data_list = []
        unknown_class_counter = 0
        for x in self.data_list:
            xlabel = x.split('/')
            xlabel = xlabel[0]
            if xlabel in LABELS:
                new_data_list.append(x)
            else:
                unknown_class_counter += 1
        for i in range(unknown_class_counter):
            sample_index = randint(0, len(self.unknown_list)-1)
            new_data_list.append(self.unknown_list[sample_index])
        self.data_list = new_data_list

    def generate_silence_sample(self):
        """
        Randomly creates a silence class sample from noise samples
        """
        if self.silence_class_zeros_count < 185:
            new_sample = np.zeros(SEQ_LENGTH, dtype = np.int16)
            self.silence_class_zeros_count += 1
        else:
            selected = self.noise_list[randint(0, len(self.noise_list)-1)]
            _, sample = read(self.root_dir+'/_background_noise_/'+selected)
            start_index = randint(0, len(sample)-SEQ_LENGTH)
            new_sample = sample[start_index:start_index+SEQ_LENGTH] * np.random.uniform(0, 1)
        new_sample = new_sample.astype(np.float32)
        return new_sample

    def add_noise_snr(self, sample):
        """
        Add noise randomly selected from provided noise in audio/_background_noise_ to a set Signal to Noise Ratio (SNR)
        SNR level is expressed in dB and randomly selected from [-5, 0, 5, 10, None]
        """
        _, noise =  read(self.root_dir+'/_background_noise_/'+ self.noise_list[randint(0, len(self.noise_list)-1)])
        start_index = randint(0, len(noise)-SEQ_LENGTH)
        noise = noise[start_index:start_index+SEQ_LENGTH]

        levels = [-5, 0, 5, 10, None]
        snr_target = levels[randint(0, len(levels)-1)]

        if snr_target is None:
            return sample
        else:
            sample_power = np.sum((sample / 2**15)**2) / len(sample)
            noise_power = np.sum((noise / 2**15)**2) / len(noise)
            factor = np.sqrt((sample_power / noise_power) / (10**(snr_target / 10.0)))
            return np.int16(sample + factor * noise)
    
    def add_noise_uniform(self, sample, upper_bound):
        """
        Add noise randomly selected from provided noise in audio/_background_noise_
        Scales the noise down by a factor randomly sampled between 0 and upper_bound.
        This method has less physical meaning but turns out to work better.
        ref: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/46839
        """
        _, noise =  read(self.root_dir+'/_background_noise_/'+ self.noise_list[randint(0, len(self.noise_list)-1)])
        start_index = randint(0, len(noise)-SEQ_LENGTH)
        noise = noise[start_index:start_index+SEQ_LENGTH]
        return np.int16(sample + np.random.uniform(0, upper_bound) * noise)
    
    def time_stretching(self, sample, range):
        """
        Shift the audio clip, I used 4800 for range as recommended in:
        ref: https://www.kaggle.com/haqishen/augmentation-methods-for-audio
        """
        shift = randint(-range, range)
        if shift >= 0:
            return np.int16(np.concatenate((sample[shift:], np.random.randint(-32, 32, shift))))
        else:
            return np.int16(np.concatenate((np.random.randint(-32, 32, -shift), sample[:shift])))
    
    def speed_tuning(self, sample):
        """
        Fasten or slow down the audio clip
        ref: https://www.kaggle.com/haqishen/augmentation-methods-for-audio
        """
        speed_rate = np.random.uniform(0.7,1.3)
        sample = sample.astype(float)
        f_sample = cv2.resize(sample, (1, int(len(sample) * speed_rate))).squeeze()
        if len(f_sample) < SEQ_LENGTH:
            pad_len = SEQ_LENGTH - len(f_sample)
            f_sample = np.r_[np.random.randint(-32, 32, int(pad_len/2)),
                                f_sample,
                                np.random.randint(-32, 32,int(np.ceil(pad_len/2)))]
            return np.int16(f_sample)
        else:
            cut_len = len(f_sample) - SEQ_LENGTH
            f_sample = f_sample[int(cut_len/2):int(cut_len/2)+SEQ_LENGTH]
            return np.int16(f_sample)

    def pitch_shifting(self, sample):
        """
        Modify the pitch of the audio
        ref: https://arxiv.org/pdf/1608.04363.pdf 
        """
        levels = [-2, -1, 1, 2, None]
        pitch_target = levels[randint(0, len(levels)-1)]
        if pitch_target is None:
            return sample
        else:
            return np.int16(pitch_shift(sample.astype(float), SEQ_LENGTH, n_steps = pitch_target))

    def reduce_dataset(self, class_size):
        """
        Function provided for convenience when debugging, reduces the dataset to 'class_size' number of sample per class.
        """
        class_distribution = np.zeros((12), dtype = np.int16)
        new_data_list = []
        for x in self.data_list:
            xlabel = x.split('/')
            xlabel = xlabel[0]
            if xlabel in LABELS:
                if class_distribution[LABELS.index(xlabel)] < class_size:
                    new_data_list.append(x)
                    class_distribution[LABELS.index(xlabel)] += 1
            else:
                if class_distribution[10] < class_size:
                    new_data_list.append(x)
                    class_distribution[10] += 1
        self.data_list = new_data_list
    
    def display(self):
        """
        Function provided for convenience, displays the number of sample per class (checks for loading issues)
        """
        class_distribution = np.zeros((12), dtype = np.int16)
        for x in self.data_list:
            xlabel = x.split('/')
            xlabel = xlabel[0]
            if xlabel in LABELS:
                class_distribution[LABELS.index(xlabel)] += 1
            else:
                class_distribution[10] += 1
        print('class distribution :  ', [(LABELS[i], class_distribution[i]) for i in range(12)])

"""
#DEBUGGING PURPOSES
data_path = '/home/r2d9/Desktop/Data/train'
dataset = dataset(data_path + '/training_list.txt', data_path + '/audio', "submission")
print(dataset[0]['label'])
"""


