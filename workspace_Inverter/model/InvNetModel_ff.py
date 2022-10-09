import torch
import torch.nn as nn
import torch.nn.functional as F

class pub(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pub, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(True))
        self.conv_layer2 = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(True))
        self.conv_layer3 = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        c1 = self.conv_layer1(x)
        c2 = self.conv_layer2(c1)
        c3 = self.conv_layer3(c2)
        return torch.cat((c1, c2, c3), dim=1)

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        self.pub = pub(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x,self.pool(x)

class up(nn.Module):
    def __init__(self, in_channels, out_channels, sample=False):
        super(up, self).__init__()
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.pub = pub(in_channels//2+out_channels, out_channels)

    def forward(self, x, x1):
        x = self.sample(x)
        x = torch.cat((x, x1), dim=1)
        x = self.pub(x)
        return x

class InvNetModel_ff(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, f_num = [8, 16, 32, 64, 128], n=3, sample=False):
        super(InvNetModel_ff, self).__init__()
        self.down1 = down(in_channels, f_num[0])
        self.down2 = down(f_num[0]*n, f_num[1])
        self.down3 = down(f_num[1]*n, f_num[2])
        self.down4 = down(f_num[2]*n, f_num[3])
        self.bridge = pub(f_num[3]*n, f_num[4])
        self.up4 = up(f_num[4]*n, f_num[3], sample)
        self.up3 = up(f_num[3]*n, f_num[2], sample)
        self.up2 = up(f_num[2]*n, f_num[1], sample)
        self.up1 = up(f_num[1]*n, f_num[0], sample)
        self.con_last = nn.Conv3d(f_num[0]*3, out_channels, 1)

    def forward(self, x):
        x1,x = self.down1(x)
        x2,x = self.down2(x)
        x3,x = self.down3(x)
        x4,x = self.down4(x)
        x = self.bridge(x)
        x = self.up4(x,x4)
        x = self.up3(x,x3)
        x = self.up2(x,x2)
        x = self.up1(x,x1)
        out = self.con_last(x)
        return out
