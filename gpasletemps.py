import numpy as np

def newSignalLength(signalLength, sizeKernel, stride, padding):
    print(((signalLength-sizeKernel+2*padding)/stride)+1)
    return ((signalLength-sizeKernel+2*padding)/stride)+1

def whichPadding(inLen, outLen, sKernel, stride):
    print(0.5*((outLen-1)*stride-inLen+sKernel))
    return 0.5*((outLen-1)*stride-inLen+sKernel)

### 1st convolution ###
#whichPadding(16000, 8000, 80, 4)
#newSignalLength(16000, 80, 4, 38)

### maxpool ###
#newSignalLength(4000, 3, 2, 20)
#whichPadding(4000, 2000, 40, 2)

### avgpool ###
#newSignalLength(250, 7, 1, 0)
#whichPadding(250, 250, 7, 1)