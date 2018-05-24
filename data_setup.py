from os import listdir
from os.path import isfile, join
import scipy.io.wavfile as scwav
import numpy as np
import acoustics
from random import randint, uniform

def make_training_list():
    onlyfiles = listdir('../Data/train/audio')
    onlyfiles.remove('_background_noise_')

    allfiles = []
    for directory in onlyfiles:
        allfiles += [directory+'/'+f for f in listdir('../Data/train/audio/'+directory) if isfile(join('../Data/train/audio/'+directory, f))]

    validation_list = open('../Data/train/validation_list.txt','r')
    testing_list = open('../Data/train/testing_list.txt','r')
    train_file = open('../Data/train/training_list.txt', 'w')

    validation = validation_list.readlines()
    validation = [x.strip() for x in validation]
    test = testing_list.readlines()
    test = [x.strip() for x in test]
    notTrain = validation + test

    for filename in allfiles:
        if filename not in notTrain:
            train_file.write(filename+'\n')


def silence_generator():
    path = '../Data/train/audio/_background_noise_'
    nsamples = 1700
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
        #randomly lower the intensity
        decrease = uniform(0, 0.6)
        new_sample = decrease*new_sample
        new_sample = np.rint(new_sample).astype('int16')
        #write file
        scwav.write('../Data/train/audio/silence/silent'+str(i), 16000, new_sample)

#silence_generator()

def append_silence(filename):
    with open(filename, 'a') as myfile:
        noise_list = [f for f in listdir('../Data/train/audio/silence') if isfile(join('../Data/train/audio/silence', f))]
        for i in range(1400):
            myfile.write('silence/'+noise_list[i]+'.wav'+'\n')

#append_silence('../Data/train/training_list.txt')



   




