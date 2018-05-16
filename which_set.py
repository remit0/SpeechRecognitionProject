import re
import os
import hashlib
from os import listdir
from os.path import isfile, join
import numpy as np

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
 
  hash_name = re.sub(r'_nohash_.*$', '', base_name)

  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.

  hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result

def make_set_files():
  files = []
  directories = listdir('../data/train/audio')
  for i in range(len(directories)):
    if directories[i] != '_background_noise_':
      files += ['../data/train/audio/'+directories[i]+'/'+f+' '+directories[i] 
              for f in listdir('../data/train/audio/'+directories[i]) 
              if isfile(join('../data/train/audio/'+directories[i], f))]

  train_file = open('../data/train/train_set.txt', 'w')
  validation_file = open('../data/train/validation_set.txt', 'w')
  test_file = open('../data/train/test_set.txt', 'w')

  for file in files:
    if which_set(file, 20, 10) == 'training':
      train_file.write(file+'\n')
    elif which_set(file, 20, 10) == 'validation':
      validation_file.write(file+'\n')
    elif which_set(file, 20, 10) == 'testing':
      test_file.write(file+'\n')

def read_set_files():
  #open document
  train_file = open('../data/train/train_set.txt', 'r')
  validation_file = open('../data/train/validation_set.txt', 'r')
  test_file = open('../data/train/test_set.txt', 'r')
  #read document
  train = train_file.readlines()
  validation = validation_file.readlines()
  test = test_file.readlines()
  # write lists
  train_list = [x.strip() for x in train]
  validation_list = [x.strip() for x in validation]
  test_list = [x.strip() for x in test]

  return train_list+validation_list+test_list

def read_set_file(setname='training'):
  if setname == 'training':
    train_file = open('../data/train/train_set.txt', 'r')
    train = train_file.readlines()
    train_list = []
    for x in train:
      path, label = x.split(' ')
      label = label.strip()
      train_list.append([path, label])
    return train_list
  elif setname == 'validation':
    validation_file = open('../data/train/validation_set.txt', 'r')
    validation = validation_file.readlines()
    validation_list = []
    for x in validation:
      path, label = x.split(' ')
      label = label.strip()
      validation_list.append([path, label])
    return validation_list
  elif setname == 'testing':
    test_file = open('../data/train/test_set.txt', 'r')
    test = test_file.readlines()
    test_list = []
    for x in test:
      path, label = x.split(' ')
      label = label.strip()
      test_list.append([path, label])
    return test_list

def label2vector(label):
  labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
  vlabel = np.zeros((12, ))
  if label in labels:
    vlabel[labels.index(label)] = 1
  else:
    vlabel[10] = 1
  return vlabel
