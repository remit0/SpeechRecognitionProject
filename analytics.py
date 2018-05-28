import matplotlib.pyplot as plt

def lossDisplay(filename):
    with open(filename,'r') as lossData:
        losses = [float(x.strip()) for x in lossData.readlines()]
    steps = [i for i in range(len(losses))]
    plt.plot(steps, losses)
    plt.show()
lossDisplay('../Data/results/monitoring/Xperience.txt')
#lossDisplay('../Data/results/monitoring/loss_step_1.txt')
#lossDisplay('../Data/results/monitoring/loss_step_2.txt')
#lossDisplay('../Data/results/monitoring/loss_step_3.txt')
