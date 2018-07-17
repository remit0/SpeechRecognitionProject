from os import listdir
from os.path import isfile, join
import scipy.io.wavfile as scwav
import numpy as np
from random import randint, uniform

"""
Functions below are run once to setup:
-> training list, validation and testing list.
-> static samples for 'silence' class for the validation and testing list.
"""

def make_training_list(filepath):
    """
    Every sample that is not in either testing or validation lists belongs to the training list
    """
    onlyfiles = listdir(filepath+'/audio')
    onlyfiles.remove('_background_noise_')

    allfiles = []
    for directory in onlyfiles:
        allfiles += [directory+'/'+f for f in listdir(filepath+'/audio/'+directory) 
        if isfile(join(filepath+'/audio/'+directory, f))]

    validation_list = open(filepath+'/validation_list.txt','r')
    testing_list = open(filepath+'/testing_list.txt','r')
    train_file = open(filepath+'/training_list.txt', 'w')

    validation = validation_list.readlines()
    validation = [x.strip() for x in validation]
    test = testing_list.readlines()
    test = [x.strip() for x in test]
    notTrain = validation + test

    for filename in allfiles:
        if filename not in notTrain:
            train_file.write(filename+'\n')

#make_training_list('../Data/train')

def silence_generator():
    path = '../Data/train/audio/_background_noise_'
    nsamples = 270
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
        #decrease = uniform(0, 0.6)         # ?
        #new_sample = decrease*new_sample
        new_sample = np.rint(new_sample).astype('int16')
        #write file
        scwav.write('../Data/train/audio/silence/silent'+str(i)+'valtest.wav', 16000, new_sample)

#silence_generator()

def append_silence(dirpath, filename):
    """
    Reads 'silence' class file names in a directory and writes it to a txt_file
    """
    with open(filename, 'a') as myfile:
        noise_list = [f for f in listdir(dirpath+'/audio/silence') 
        if isfile(join(dirpath+'/audio/silence', f))]  # egacy function to be changed
        for i in range(0, 270):
            myfile.write('silence/silent'+str(i)+'valtest.wav\n')

#append_silence('../Data/train', '../Data/train/testing_list.txt')
#append_silence(../Data/train, '../Data/train/test_file.txt')

def clear_silence(filename):
    """
    clear 'silence' occurences in a text file
    """
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    f = open(filename,'w')
    for line in lines:
        if 'silence' not in line:
            f.write(line)
    f.close()

#clear_silence('../Data/train/validation_list.txt')

def make_testing_list(input_path, output_path):
    all_test_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    #print(all_test_files)
    with open(output_path + '/submission_list.txt', 'w') as submission_file:
        for x in all_test_files:
            submission_file.write(x +'\n')
    
make_testing_list('../Data/test/audio/', '../Data')





   




