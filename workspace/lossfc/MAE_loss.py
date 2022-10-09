import torch
import torch.nn as nn
import numpy

class MAE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out, mask):
        return torch.mean(torch.abs(out - mask.cuda()))
