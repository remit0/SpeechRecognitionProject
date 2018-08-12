import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
# pylint: disable=E1101, W0612

"""
filter banks computations from : https://github.com/jameslyons/python_speech_features
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

def filter_banks(sample):
    signal = sample.numpy()

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

    filter_banks = torch.from_numpy(filter_banks)
    filter_banks = filter_banks.type(torch.FloatTensor)
    return filter_banks # time x features

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 3), padding = (3, 1))
        self.maxpool1 = nn.MaxPool2d((1, 3))
        self.conv2 = nn.Conv2d(64, 128, (1, 7), padding=(0, 3))
        self.maxpool2 = nn.MaxPool2d((1, 4))
        self.conv3 = nn.Conv2d(128, 256, (1, 10))
        self.conv4 = nn.Conv2d(256, 512, (7, 1), padding=(3, 0))
        self.maxpool3 = nn.MaxPool1d(98)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 12)
        
    def forward(self, x):
        with torch.no_grad():
            inx = torch.ones(x.size(0), 98, 120)
            for i in range(x.size(0)):
                inx[i, :, :] = filter_banks(x[i])
        inx = inx.unsqueeze(1).to(DEVICE)
        inx = self.conv1(inx)
        inx = self.maxpool1(inx)
        inx = self.conv2(inx)
        inx = self.maxpool2(inx)
        inx = self.conv3(inx)
        inx = self.conv4(inx)
        inx = inx.squeeze(3)
        inx = self.maxpool3(inx)
        inx = inx.squeeze(2)
        inx = self.dropout(inx)
        inx = self.fc1(inx)
        inx = self.fc2(inx)

        return inx

def accuracy(model, dataset, filename, batchsize=2):
    """
    Computes overall accuracy on the dataset provided
    """
    total, correct = 0, 0
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = False)

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'])
            _, predicted = torch.max(outputs.data, 1)
            total += batchsize
            correct += (predicted == batch['label'].to(DEVICE)).sum().item()

    with open(filename, 'a') as f:
        f.write(str(100 * correct / float(total))+'\n')
    model.train()
    return(100*correct/float(total))

def class_accuracy(model, dataset, filename, batchsize=2):
    """
    Computes per class accuracy on the dataset provided
    """
    labels = ['yes','no','up','down','left','right','on','off','stop','go','unknown','silence']
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
    model.eval()
    dataloader = DataLoader(dataset, batch_size = batchsize, drop_last = False)
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            outputs = model(batch['audio'])
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == batch['label'].to(DEVICE)).squeeze()

            for i in range(batchsize):
                label = batch['label'][i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    with open(filename, 'w') as myFile:
        for i in range(12):        
            myFile.write('Accuracy of %5s : %2d %%' % (
            labels[i], 100 * class_correct[i] / class_total[i])+'\n')
    model.train()