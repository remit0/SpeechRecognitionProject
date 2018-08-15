from os import listdir
from os.path import isfile, join
from scipy.io.wavfile import read, write
import numpy as np
from random import randint

def make_training_list(filepath):
    """
    Generates training list files which contains the path to every training sample.
    Testing and validation lists are already provided in the kaggle dataset.
    Every sample that is not in either testing or validation lists belongs to the training list.
    filepath is the path to the train directory.
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

#make_training_list('home/r2d9/Desktop/Data/train')

def silence_generator(path):
    """
    Creates silence class samples to be part of validation and testing set
    path is the path to the audio directory. It assumes that there is a silence folder under the audio folder
    """
    nsamples = 270
    bn_path = path + '/_background_noise_'
    noise_list = [f for f in listdir(bn_path) if isfile(join(bn_path, f))]
    noise_list.remove('README.md')
    
    for i in range(nsamples):
        if i < 27:
            new_sample = np.zeros(16000, dtype='int16')
        else:
            selected = noise_list[randint(0, len(noise_list)-1)]
            _, sample = read(bn_path+'/'+selected)
            start_index = randint(0, len(sample)-16000)
            new_sample = sample[start_index:start_index+16000] * np.random.uniform(0, 1)
            new_sample = np.rint(new_sample).astype('int16')
        write(path + '/silence/silent'+str(i)+'valtest.wav', 16000, new_sample)

#silence_generator('home/r2d9/Desktop/Data/train/audio')

def append_silence(dirpath):
    """
    Appends 'silence' audio clips to testing/validation lists.
    Assumes silence_generator has been used to produce the audio clips.
    Requires path to the train directory
    """
    testing_list = dirpath + '/testing_list.txt'
    validation_list = dirpath + '/validation_list.txt'
    with open(testing_list, 'a') as myfile:
        for i in range(0, 270):
            myfile.write('silence/silent'+str(i)+'valtest.wav\n')
    with open(validation_list, 'a') as myfile:
        for i in range(0, 270):
            myfile.write('silence/silent'+str(i)+'valtest.wav\n')

#append_silence('/home/r2d9/Desktop/Data/train')

def clear_silence(filepath):
    """
    Clears 'silence' occurences in a text file
    Provided for convenience in case append_silence has been wrongly used
    Requires the path to the file you wish to remove silence occurences from
    """
    f = open(filepath,'r')
    lines = f.readlines()
    f.close()
    f = open(filepath,'w')
    for line in lines:
        if 'silence' not in line:
            f.write(line)
    f.close()

#clear_silence('/home/r2d9/Desktop/Data/train/validation_list.txt')

def make_testing_list(input_path, output_path):
    """
    Creates a text file with all the files of the competition testing set (unlabelled data)
    Requires the path to the audio directory of the test set
    """
    all_test_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    with open(output_path + '/submission_list.txt', 'w') as submission_file:
        for x in all_test_files:
            submission_file.write(x +'\n')

#make_testing_list('/home/r2d9/Desktop/Data/test/audio', '/home/r2d9/Desktop/Data')

def make_training_list_complete(filepath):
    """
    full training samples
    """
    onlyfiles = listdir(filepath+'/audio')
    onlyfiles.remove('_background_noise_')

    allfiles = []
    for directory in onlyfiles:
        allfiles += [directory+'/'+f for f in listdir(filepath+'/audio/'+directory) 
        if isfile(join(filepath+'/audio/'+directory, f))]

    train_file = open(filepath+'/complete_list.txt', 'w')
    for filename in allfiles:
        train_file.write(filename+'\n')

make_training_list_complete('/home/r2d9/Desktop/Data/train')



   




