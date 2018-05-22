import numpy as np

"""
Compute the size of a vector after a convolution or any layer that can modify the vector size.
"""

def newSignalLength(signalLength, sizeKernel, stride, padding):
    print(((signalLength-sizeKernel+2*padding)/stride)+1)
    return ((signalLength-sizeKernel+2*padding)/stride)+1

def whichPadding(inLen, outLen, sKernel, stride):
    print(0.5*((outLen-1)*stride-inLen+sKernel))
    return 0.5*((outLen-1)*stride-inLen+sKernel)

