"""
TO WORK ON :
-- adding noise during training
-- kernel from forum
test best models on the real test set
3steps dilated convolution on wavenet
kernel size inside resnet blocks
encadrement learning_rate for 1st convolution // full model
dilated convolution model -> wavenet 3/4 layers
change silence class generation model // generate on the fly silence samples
batch results copy on results.xls
"""

"""
RUNNING :
try only for resnet different kernels of 1st convolution stride 2 8 16 
smaller than 1ms kernel 0.5ms
"""

"""
COMPLETED :
try only for resnet different kernels of 1st convolution 10 / 20 / 1 ms
full model in one go (dilated)
full model in one go (GRU) - more tests on 3 steps ?
step 3 on model 1
testing CNN - architecture ? spectrogram that works well for time series
"""