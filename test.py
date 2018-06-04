# pylint: disable=E1101, W0612
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)