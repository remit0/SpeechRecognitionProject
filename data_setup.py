from os import listdir
from os.path import isfile, join

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




