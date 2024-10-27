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


# class Bottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=4):
#         super(Bottleneck, self).__init__()
#         mid_channels = in_channels // reduction
#         self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3x3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

#     def forward(self, x):
#         return self.conv3x3(self.relu(self.conv1x1(x)))
    

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


# class ResidualDenseBlock_5C(nn.Module):
#     def __init__(self, nf=64, gc=32, attn_dim=32, bias=True):
#         super(ResidualDenseBlock_5C, self).__init__()
#         self.gc = gc
#         self.nf = nf
#         self.bottleneck1 = Bottleneck(nf, gc)
#         self.bottleneck2 = Bottleneck(nf + gc, gc)
#         self.bottleneck3 = Bottleneck(nf + 2 * gc, gc)
#         self.bottleneck4 = Bottleneck(nf + 3 * gc, gc)
#         self.bottleneck5 = nn.Sequential(
#             nn.Conv2d(4 * gc, 4 * gc, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(4 * gc, 2 * gc, kernel_size=3, padding=1)
#         )
#         self.final_conv = nn.Conv2d(nf , nf, kernel_size=3, padding=1, bias=bias)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         # Initialize MultiheadAttention with correct embed_dim
#         self.attn = nn.MultiheadAttention(embed_dim=gc, num_heads=8, dropout=0.1)

#     def forward(self, x):
#         x1 = self.bottleneck1(x)
#         x2 = self.bottleneck2(torch.cat((x, x1), 1))
#         x3 = self.bottleneck3(torch.cat((x, x1, x2), 1))
#         x4 = self.bottleneck4(torch.cat((x, x1, x2, x3), 1))
#         x_cat = self.bottleneck5(torch.cat((x1, x2, x3, x4), 1))
        
#         b, c, h, w = x_cat.shape
#         # print(f"Shape before rearrange: {x_cat.shape}")
#         # print(f"gc: {self.gc}")

#         # Calculate the expected number of channels after rearrange
#         n = 2  # number of concatenated inputs in x_cat
#         new_c = c // n
#         # print(f"new_c: {new_c}")

#         # Ensure correct rearrangement
#         # Use the appropriate pattern to match the target shape for MultiheadAttention
#         x_cat = rearrange(x_cat, 'b (n c) h w -> (b h w) n c', n=n, c=new_c, h=h, w=w)
#         x_cat = x_cat.permute(1, 0, 2)  # (n, b*h*w, c) -> (n, b*h*w, c)
        
#         # Ensure the embed_dim matches gc
#         assert self.attn.embed_dim == self.gc, f"Expected embed_dim={self.gc}, but got {self.attn.embed_dim}."

#         x_attn, _ = self.attn(x_cat, x_cat, x_cat)  # Self-attention
#         x_attn = x_attn.permute(1, 0, 2).reshape(b, c, h, w)  # Reshape back
#         # print(x_attn.shape)   torch.Size([4, 64, 64, 64])
#         # print(self.gc) 32
#         # print(self.nf)  64
#         x5 = self.final_conv(x_attn)
#         return x5 * 0.2 + x




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
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)#残差密集块
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)#RRDB块序列之后的卷积层，用于进一步处理特征。
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        print(f'Input to RRDBNet: {x.shape}')  # Debugging line torch.Size([4, 3, 64, 64])
        fea = self.conv_first(x)
        # print(f'After conv_first: {fea.shape}')  # Debugging line torch.Size([4, 64, 64, 64])
        
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        # print(f'After RRDB_trunk: {trunk.shape}')  # Debugging line  torch.Size([4, 64, 64, 64])
        
        fea = fea + trunk
        # print(f'After trunk addition: {fea.shape}')  # Debugging line  torch.Size([4, 64, 64, 64])
        # aa = F.interpolate(fea, scale_factor=2, mode='nearest') 
        # print(f'After upconv1: {aa.shape}')  # Debugging line torch.Size([4, 64, 128, 128])
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # print(f'After upconv1: {fea.shape}')  # Debugging line torch.Size([4, 64, 128, 128])
        
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # print(f'After upconv2: {fea.shape}')  # Debugging line torch.Size([4, 64, 256, 256])
        
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        # print(f'After conv_last: {out.shape}')  # Debugging line torch.Size([4, 3, 256, 256])

        return out
