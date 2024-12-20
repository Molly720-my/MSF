import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction

from torch import nn, einsum
from einops import rearrange, repeat


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.init_weights()

    def init_weights(self):
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i+1}'), 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        # print(f'After conv1: {x1.shape}')  # Debugging line torch.Size([4, 32, 64, 64])
        
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # print(f'After conv2: {x2.shape}')  # Debugging line torch.Size([4, 32, 64, 64])
        
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # print(f'After conv3: {x3.shape}')  # Debugging line torch.Size([4, 32, 64, 64])
        
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # print(f'After conv4: {x4.shape}')  # Debugging line torch.Size([4, 32, 64, 64])
        
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # print(f'After conv5: {x5.shape}')  # Debugging line torch.Size([4, 64, 64, 64])
        
        return x5 * 0.2 + x


def default_init_weights(module, scale=1):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            m.weight.data *= scale


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        # print(f'Input to RRDB: {x.shape}')  # Debugging line torch.Size([4, 64, 64, 64])
        out = self.RDB1(x)
        # print(f'After RDB1: {out.shape}')  # Debugging line torch.Size([4, 64, 64, 64])
        
        out = self.RDB2(out)
        # print(f'After RDB2: {out.shape}')  # Debugging line torch.Size([4, 64, 64, 64])
        
        out = self.RDB3(out)
        # print(f'After RDB3: {out.shape}')  # Debugging line torch.Size([4, 64, 64, 64])
        
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)#
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)#
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        print(f'Input to RRDBNet: {x.shape}')  # Debugging line torch.Size([4, 3, 64, 64])
        fea = self.conv_first(x)       
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # print(f'After upconv1: {fea.shape}')  # Debugging line torch.Size([4, 64, 128, 128])
        
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # print(f'After upconv2: {fea.shape}')  # Debugging line torch.Size([4, 64, 256, 256])
        
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        # print(f'After conv_last: {out.shape}')  # Debugging line torch.Size([4, 3, 256, 256])

        return out
