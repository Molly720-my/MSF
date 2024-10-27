'''
Codes are inherited from:
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py
Modified by Bingchen Li
'''


from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from torch.nn import Softmax
from models.mambalay import MambaLayer
from models.fusion_mamba import feature_fusion
# from mamba_ssm import Mamba
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, context):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=context, out_channels=in_dim//16, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        
        proj_value = self.value_conv(y)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        
        # Ensure out_H and out_W have the same size as x
        out_H = F.interpolate(out_H, size=(height, width), mode='bilinear', align_corners=False)
        out_W = F.interpolate(out_W, size=(height, width), mode='bilinear', align_corners=False)
        
        return self.gamma * (out_H + out_W) + x
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        # query: semantic; key, value: image.

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = nn.Parameter(torch.tensor(dim_head ** -0.5))  # 可学习
        # self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, context=None, mask=None):
        h = self.heads

        # Projecting inputs
        # print(f'Query shape: {x.shape}') # torch.Size([4, 256, 64])
        q = self.to_q(x)
        # print(f'Query shape: {q.shape}') torch.Size([4, 256, 64])
        
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # print(f'Key shape: {k.shape}') torch.Size([4, 256, 64])
        # print(f'Value shape: {v.shape}') torch.Size([4, 256, 64])

        # Reshaping for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print(f'Reshaped Query shape: {q.shape}') torch.Size([4, 256, 64])
        # print(f'Reshaped Key shape: {k.shape}')  torch.Size([4, 256, 64])
        # print(f'Reshaped Value shape: {v.shape}')  torch.Size([4, 256, 64])

        # Compute attention scores
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # print(f'Similarity shape: {sim.shape}') torch.Size([4, 256, 256])

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # Compute attention weights
        attn = sim.softmax(dim=-1)
        # print(f'Attention shape: {attn.shape}') torch.Size([4, 256, 256])

        # Compute output
        out = einsum('b i j, b j d -> b i d', attn, v)
        # print(f'Output shape after attention: {out.shape}')  torch.Size([4, 256, 64])

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # print(f'Output shape after rearrange: {out.shape}')  torch.Size([4, 256, 64])

        out = self.to_out(out)
        # print(f'Final Output shape: {out.shape}')  torch.Size([4, 256, 64])

        return out
    
    # def forward(self, x, context=None, mask=None):
    #     h = self.heads

    #     q = self.to_q(x)
    #     context = default(context, x)
    #     k = self.to_k(context)
    #     v = self.to_v(context)

    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    #     sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    #     if exists(mask):
    #         mask = rearrange(mask, 'b ... -> b (...)')
    #         max_neg_value = -torch.finfo(sim.dtype).max
    #         mask = repeat(mask, 'b j -> (b h) () j', h=h)
    #         sim.masked_fill_(~mask, max_neg_value)

    #     # attention, what we cannot get enough of
    #     attn = sim.softmax(dim=-1)

    #     out = einsum('b i j, b j d -> b i d', attn, v)
    #     out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    #     return self.to_out(out)
from einops.layers.torch import Rearrange
class AttentionPool(nn.Module):
    def __init__(self, input_dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Parameter(torch.eye(input_dim))

    def forward(self, x):
        _, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = x[:,:,:-remainder]

        attn_logits = einsum('b d n, d e -> b e n', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # x = self.attn1(self.norm1(x)) + x
        # x = self.attn2(self.norm2(x), context=context) + x
        # x = self.ff(self.norm3(x)) + x
        x = self.attn1(self.norm1(x))
        x = self.attn2(self.norm2(x), context=context)
        x = self.ff(self.norm3(x))
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # x = self.attn1(self.norm1(x)) + x
        # x = self.attn2(self.norm2(x), context=context) + x
        # x = self.ff(self.norm3(x)) + x
        x = self.attn1(self.norm1(x))
        x = self.attn2(self.norm2(x), context=context)
        x = self.ff(self.norm3(x))
        return x



   
class GradualTransition(nn.Module):
    def __init__(self, in_channels, mid_channels1, mid_channels2, out_channels):
        super(GradualTransition, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels1, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels1, mid_channels2, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(mid_channels2, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)       
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        
        out = self.conv1(x)
        # print(f"After Conv1 (1x1): {out.shape}")
        out = self.bn1(out)
        # print(f"After BatchNorm1: {out.shape}")
        out = self.relu1(out)
        # print(f"After ReLU1: {out.shape}")
        
        out = self.conv2(out)
        # print(f"After Conv2 (1x1): {out.shape}")
        out = self.bn2(out)
        # print(f"After BatchNorm2: {out.shape}")
        out = self.relu2(out)
        # print(f"After ReLU2: {out.shape}")
        
        out = self.conv3(out)
        # print(f"After Conv3 (1x1): {out.shape}")
        out = self.bn3(out)
        # print(f"After BatchNorm3: {out.shape}")

        out_sk = self.conv4(x)
        out_sk = self.bn3(out_sk)

        out += out_sk

        return out

 
class GradualTransition_512(nn.Module):
    def __init__(self, in_channels, mid_channels2, out_channels):
        super(GradualTransition_512, self).__init__()
        
      
        
        self.conv2 = nn.Conv2d(in_channels, mid_channels2, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(mid_channels2, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)       
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        
     
        
        out = self.conv2(x)
        # print(f"After Conv2 (1x1): {out.shape}")
        out = self.bn2(out)
        # print(f"After BatchNorm2: {out.shape}")
        out = self.relu2(out)
        # print(f"After ReLU2: {out.shape}")
        
        out = self.conv3(out)
        # print(f"After Conv3 (1x1): {out.shape}")
        out = self.bn3(out)
        # print(f"After BatchNorm3: {out.shape}")

        out_sk = self.conv4(x)
        out_sk = self.bn3(out_sk)

        out += out_sk

        return out



# class ModifiedSpatialTransformer(nn.Module):
#     """
#     Transformer block for image-like data.
#     First, project the input (aka embedding)
#     and reshape to b, t, d.
#     Then apply standard transformer action.
#     Finally, reshape to image
#     """
#     def __init__(self, in_channels, n_heads, d_head,
#                  depth=1, dropout=0., context_dim=1024, up_factor=2, is_last=False):
#         super().__init__()
#         self.in_channels = in_channels
#         inner_dim = n_heads * d_head
#         self.norm = Normalize(in_channels)


#         self.proj_in = nn.Conv2d(in_channels,
#                                  inner_dim,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.proj_context = nn.Conv2d(context_dim,
#                                  inner_dim,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)

#         self.transformer_blocks = nn.ModuleList(
#             [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=inner_dim)
#                 for d in range(depth)]
#         )


#         self.proj_out = zero_module(nn.Conv2d(inner_dim,
#                                               in_channels,
#                                               kernel_size=1,
#                                               stride=1,
#                                               padding=0))
#         up_channels = int(in_channels / up_factor / up_factor)
#         if not is_last:
#             self.conv_out = nn.Conv2d(up_channels, up_channels, 3, 1, 1)
#         else:
#             self.conv_out = nn.Conv2d(up_channels, up_channels, 4, 1, 1)
#         self.up_factor = up_factor


#     def forward(self, x, context=None):
#         # note: if no context is given, cross-attention defaults to self-attention
#         b, c, h, w = x.shape
#         # print(f'Input shape: {x.shape}') torch.Size([4, 1024, 16, 16])
        
#         x = self.norm(x)
#         # print(f'After normalization: {x.shape}') torch.Size([4, 1024, 16, 16])
        
#         x = self.proj_in(x)
#         # print(f'After proj_in: {x.shape}')  torch.Size([4, 64, 16, 16])
        
#         context = self.proj_context(context)
#         # print(f'Context after proj_context: {context.shape}')  torch.Size([4, 64, 64, 64])
        
#         x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
#         # print(f'After rearrange x: {x.shape}')  torch.Size([4, 256, 64])
        
#         context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
#         # print(f'After rearrange context: {context.shape}')  torch.Size([4, 4096, 64])
        
#         for block in self.transformer_blocks:
#             x = block(x, context=context)
#             # print(f'After transformer block: {x.shape}') torch.Size([4, 256, 64])
        
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
#         # print(f'After rearrange x back to image: {x.shape}')  torch.Size([4, 64, 16, 16])
        
#         x = self.proj_out(x)
#         # print(f'After proj_out: {x.shape}')  torch.Size([4, 1024, 16, 16])
        
#         x = rearrange(x, 'b (c uw uh) h w -> b c (h uh) (w uw)', uh=self.up_factor, uw=self.up_factor).contiguous()
#         # print(f'After final rearrange: {x.shape}') torch.Size([4, 64, 64, 64])
        
#         x = self.conv_out(x)
#         # print(f'After conv_out: {x.shape}') torch.Size([4, 64, 64, 64])
        
#         return x
    

class ModifiedSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=1024, up_factor=2, is_last=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        # new
        self.inverted_residual_in = GradualTransition(1024, 512, 128, 64)
        
        # self.proj_in = nn.Conv2d(in_channels,
        #                          inner_dim,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        self.proj_context = nn.Conv2d(context_dim,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_context_ = GradualTransition_512(512, 128, 64)

        # self.transformer_blocks = nn.ModuleList(
        #     [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=inner_dim)
        #         for d in range(depth)]
        # )
        # new
        self.inverted_residual_out = GradualTransition(64, 256, 512, 1024)

        # self.proj_out = zero_module(nn.Conv2d(inner_dim,
        #                                       in_channels,
        #                                       kernel_size=1,
        #                                       stride=1,
        #                                       padding=0))
        up_channels = int(in_channels / up_factor / up_factor)
        if not is_last:
            self.conv_out = nn.Conv2d(up_channels, up_channels, 3, 1, 1)
        else:
            self.conv_out = nn.Conv2d(up_channels, up_channels, 4, 1, 1)
        self.up_factor = up_factor
        self.mamba1 = MambaLayer(dim=64).cuda()
        self.mamba2 = MambaLayer(dim=1024).cuda()
        self.mamba3 = MambaLayer(dim=256).cuda()

    def forward(self, x, context=None):
        
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        # print(f'Input shape: {x.shape}') torch.Size([4, 1024, 16, 16])
        # print(f'Input shape: {x.shape}') 
        x = self.norm(x)
        # print(f'After normalization: {x.shape}') torch.Size([4, 1024, 16, 16])
        x = self.inverted_residual_in(x)
        # print(f'After inverted_residual_in: {x.shape}') torch.Size([4, 64, 16, 16])
        x = self.mamba1(x)
        # x = self.proj_in(x)
        # print(f'After proj_in: {x.shape}')  torch.Size([4, 64, 16, 16])
        # print(f'After proj_in: {x.shape}') 
        # print(f'Context : {context.shape}')
        if c== 512:
            context = self.proj_context_(context)
        else:
            context = self.proj_context(context)
        context = self.mamba1(context)
        # print(f'Context after proj_context: {context.shape}')  torch.Size([4, 64, 64, 64])
        # print(f'Context after proj_context: {context.shape}')
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        # print(f'After rearrange x: {x.shape}')  #torch.Size([4, 256, 64])
        
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
        # print(f'After rearrange context: {context.shape}') # torch.Size([4, 4096, 64])
        
        # for block in self.transformer_blocks:
        #     x = block(x, context=context)
        #     print(f'After transformer block: {x.shape}') #torch.Size([4, 256, 64])
        x = feature_fusion(x,context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        # print(f'After rearrange x back to image: {x.shape}')  torch.Size([4, 64, 16, 16])
        # print(f'After rearrange x back to image: {x.shape}') 
        x = self.inverted_residual_out(x)
        x = self.mamba2(x)
        # x = self.proj_out(x)
        # print(f'After proj_out: {x.shape}') 
        # print(f'After proj_out: {x.shape}')  torch.Size([4, 1024, 16, 16])
        
        x = rearrange(x, 'b (c uw uh) h w -> b c (h uh) (w uw)', uh=self.up_factor, uw=self.up_factor).contiguous()
        # print(f'After final rearrange: {x.shape}') torch.Size([4, 64, 64, 64])
        
        x = self.conv_out(x)
        # x = self.mamba1(x)
        # print(f'After conv_out: {x.shape}') torch.Size([4, 64, 64, 64])
        
        return x



class CausalPrefixAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        max_heads_process=2,
        dropout=0.0,
        cross_attn_dropout=0.0
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.max_heads_process = max_heads_process

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.cross_attn_dropout = cross_attn_dropout  # they drop out a percentage of the prefix during training, shown to help prevent overfitting

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, context_mask=None):
        batch, context_len, device = x.shape[0], context.shape[-2], x.device

        # take care of cross attention dropout
        if self.training and self.cross_attn_dropout > 0.0:
            rand = torch.zeros((batch, context_len), device=device).uniform_()
            keep_context_len = context_len - int(context_len * self.cross_attn_dropout)
            keep_indices = rand.topk(keep_context_len, dim=-1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()

            context = rearrange(context[keep_mask], "(b n) d -> b n d", b=batch)

            if exists(context_mask):
                context_mask = rearrange(
                    context_mask[keep_mask], "(b n) -> b n", b=batch
                )

        # normalization
        x = self.norm(x)
        context = self.context_norm(context)

        # derive queries, keys, values
        q = self.to_q(x)

        k_input, v_input = self.to_kv(x).chunk(2, dim=-1)
        k_context, v_context = self.to_kv(context).chunk(2, dim=-1)

        k = torch.cat((k_context, k_input), dim=1)
        v = torch.cat((v_context, v_input), dim=1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        q = q * self.scale

        # take care of masking
        i, j = q.shape[-2], k.shape[-2]
        mask_value = -torch.finfo(q.dtype).max

        if exists(context_mask):
            mask_len = context_mask.shape[-1]
            context_mask = F.pad(context_mask, (0, max(j - mask_len, 0)), value=True)
            context_mask = rearrange(context_mask, "b j -> b 1 1 j")

        causal_mask = torch.ones((i, j), device=x.device, dtype=torch.bool).triu(
            j - i + 1
        )

        # process in chunks of heads
        out = []
        max_heads = self.max_heads_process
        for q_chunk, k_chunk, v_chunk in zip(
            q.split(max_heads, dim=1),
            k.split(max_heads, dim=1),
            v.split(max_heads, dim=1),
        ):
            sim = einsum("b h i d, b h j d -> b h i j", q_chunk, k_chunk)

            if exists(context_mask):
                sim = sim.masked_fill(~context_mask, mask_value)

            sim = sim.masked_fill(causal_mask, mask_value)

            attn = sim.softmax(dim=-1)
            attn = self.dropout(attn)

            out_chunk = einsum("b h i j, b h j d -> b h i d", attn, v_chunk)
            out.append(out_chunk)

        # concat all the heads together
        out = torch.cat(out, dim=1)

        # merge heads and then combine with linear
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)
    
    
if __name__ == '__main__':
    M = ModifiedSpatialTransformer(1024, n_heads=1, d_head=64, context_dim=128)
    img = torch.randn(1, 128, 32, 32)
    context = torch.randn(1, 1024, 16, 16)

    haha = M(context, img)
    print(haha.size())

    a = torch.randn(1, 9, 2, 2)
    l = nn.PixelShuffle(3)
    b = l(a)
    print(b)
    c = rearrange(a, 'b (c uh uw) h w -> b c (h uh) (w uw)', uh=3, uw=3)
    print(b==c)