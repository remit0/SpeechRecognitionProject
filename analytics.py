import matplotlib.pyplot as plt

NBATCH = 1112

def lossDisplay(filename):
    with open(filename,'r') as lossData:
        losses = [float(x.strip()) for x in lossData.readlines()]
    length = len(losses)
    print(length)
    nepoch = (length // NBATCH) + 1
    combien = nepoch - 1

    x = [i for i in range(nepoch-1)]
    y = [[] for i in range(combien)]
    for i in range(combien):
        y[i] = losses[i*NBATCH:(i+1)*NBATCH]
    
    y = [sum(y[i])/len(y[i]) for i in range(combien)]
    
    if length%NBATCH != 0:
        x.append(nepoch-1)
        ylast = losses[length-(length%NBATCH):]
        print(ylast)
        y.append(sum(ylast)/len(ylast))
    
    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

lossDisplay('../aliBaba/loss_step_1.txt')

#lossDisplay('../Data/results/monitoring/loss_step_2.txt')
#lossDisplay('../Data/results/monitoring/loss_step_3.txt')
