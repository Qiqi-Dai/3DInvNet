import torch
import torch.nn as nn
import numpy

class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out, mask):
        return torch.mean(torch.pow((out - mask.cuda()), 2))