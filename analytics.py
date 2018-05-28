import matplotlib.pyplot as plt

def lossDisplay(filename):
    with open(filename,'r') as lossData:
        losses = [float(x.strip()) for x in lossData.readlines()]
    losses = losses[0:150]
    steps = [i for i in range(150)]
    plt.plot(steps, losses)
    plt.xlabel("batch number")
    plt.ylabel("loss")
    plt.title("loss against batch number")
    plt.show()
lossDisplay('../Data/results/monitoring/Xperience.txt')
#lossDisplay('../Data/results/monitoring/loss_step_1.txt')
#lossDisplay('../Data/results/monitoring/loss_step_2.txt')
#lossDisplay('../Data/results/monitoring/loss_step_3.txt')
