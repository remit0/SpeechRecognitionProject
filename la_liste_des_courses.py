"""
TO WORK ON :
-- stretching shifting (in addition)
"""
"""
RUNNING :
-- adding noise during training + scale down uniformly 0.1 + scale down uniformly 1
"""
"""
COMPLETED :
silent class remodelling: 10% as pure silent / scale down silence uniformly law [0, 1]
try kernel 5ms - stride 1 ms - kernel 9 and 15 train full model one go (model1) + submit
try kernel 5ms - stride 0.25 ms - kernel 9 train full model one go + submit
submit results on key 58 and 56
--
change early stopping to 10 (all models)
kernel size inside resnet blocks (model1 - step 1)
test best models on the real test set (all models)
try only for resnet different kernels of 1st convolution stride 2 8 16  (model1 - step 1)
per class accuracy (all models)
smaller kernel 0.5ms (model1 - step 1)
--
try only for resnet different kernels of 1st convolution 10 / 20 / 1 ms (model1 - step 1)
full model in one go (model5)
full model in one go (model1)
training step 3 (model1)
testing spec CNN (model4)
"""