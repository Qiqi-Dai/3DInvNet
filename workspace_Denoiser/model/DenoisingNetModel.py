import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        layers = [
                    nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                 ]
        self.conv_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layer(x)

class res_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_conv_block, self).__init__()
        layers = [
                    nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1),
                 ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        out = self.body(x)
        return F.relu(out+x)

class attention_block(nn.Module):
    def __init__(self, channels, reduction):
        super(attention_block, self).__init__()
        self.ave_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias = False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.ave_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class DenoisingNetModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, f_num=8, reduction=2):
        super(DenoisingNetModel, self).__init__()
        self.feature_extraction = conv_block(in_channels, f_num)
        self.res_net1 = res_conv_block(f_num, f_num)
        self.res_net2 = res_conv_block(f_num, f_num)
        self.feature_attention1 = attention_block(f_num, reduction)
        self.res_net3 = res_conv_block(f_num, f_num)
        self.res_net4 = res_conv_block(f_num, f_num)
        self.feature_attention2 = attention_block(f_num, reduction)
        self.reconstruction = nn.Conv3d(f_num, out_channels, 3, 1, 1)
        self.act_layer = nn.ReLU(True)

    def forward(self, x):
        fe = self.feature_extraction(x)
        
        r1 = self.res_net1(fe)
        r2 = self.res_net2(r1)
        a1 = self.feature_attention1(r2)
        
        r3 = self.res_net3(a1)
        r4 = self.res_net4(r3)
        a2 = self.feature_attention2(r4)
        fr = self.reconstruction(fe + a2)
        return self.act_layer(x + fr)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# t = DenoisingNetModel().to(device)
# print(t)
