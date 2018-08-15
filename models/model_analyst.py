import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
# pylint: disable=E1101, W0612

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(48, 96)
        self.fc2 = nn.Linear(96, 12)
    
    def forward(self, x):
        x = x.to(DEVICE)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def accuracy(models, dataset, filename, batchsize=1):
    """
    Computes overall accuracy on the dataset provided
    """
    total, correct = 0, 0
    model1 = models[1]
    model2 = models[2]
    model3 = models[3]
    model4 = models[4]
    model_analyst = models[0]
    model_analyst.eval()

    data = DataLoader(dataset, batch_size = 1, drop_last = False)

    with torch.no_grad():
        for i_batch, batch in enumerate(data):
            outputs1 = softmax(model1(batch['audio']).squeeze(0), dim = 0)
            outputs2 = softmax(model2(batch['audio']).squeeze(0), dim = 0)
            outputs3 = softmax(model3(batch['audio']).squeeze(0), dim = 0)
            outputs4 = softmax(model4(batch['audio']).squeeze(0), dim = 0)
            
            input = torch.cat((outputs1, outputs2, outputs3, outputs4), dim = 0)
            outputs = model_analyst(input.unsqueeze(0))
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += 1
            correct += (predicted == batch['label'].to(DEVICE)).sum().item()

    with open(filename, 'a') as f:
        f.write(str(100 * correct / float(total))+'\n')
    model_analyst.train()
    return(100*correct/float(total))