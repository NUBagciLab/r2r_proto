# Taken from https://github.com/microsoft/CvT and modified

from functools import partial
from itertools import repeat
#from torch._six import container_abcs
import collections.abc as container_abcs

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
#from einops import rearrange
from einops import rearrange, repeat as e_repeat
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_

#from .registry import register_model


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w):
        if VERB: print('   ', self.__class__.__name__)
        if VERB: print('      x      ', x.shape)
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        if VERB: print('      x      ', x.shape)
        if VERB: print('      q      ', q.shape)
        if VERB: print('      k      ', k.shape)
        if VERB: print('      v      ', v.shape)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        if VERB: print('      attn   ', attn.shape)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        if VERB: print('      a * v   ', x.shape)
        x = rearrange(x, 'b h t d -> b t (h d)')
        if VERB: print('      x      ', x.shape)

        x = self.proj(x)
        if VERB: print('      x-proj ', x.shape)
        x = self.proj_drop(x)
        if VERB: print('      x-pdrop', x.shape)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
            hasattr(module, 'conv_proj_q')
            and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
            hasattr(module, 'conv_proj_k')
            and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
            hasattr(module, 'conv_proj_v')
            and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Attention_v5(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False,
                 attn_dim=-1,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        #self.conv_proj_k = self._build_projection(
        #    dim_in, dim_out, kernel_size, padding_kv,
        #    stride_kv, method
        #)
        #self.conv_proj_v = self._build_projection(
        #    dim_in, dim_out, kernel_size, padding_kv,
        #    stride_kv, method
        #)

        self.attend = nn.Softmax(dim=attn_dim)

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        #self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        #self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proto_k = nn.parameter.Parameter(torch.randn(num_heads, 1, dim_in//num_heads), requires_grad=True)
        self.proto_v = nn.parameter.Parameter(torch.randn(num_heads, 1, dim_in//num_heads), requires_grad=True)

        self.mask_addon = nn.Sequential(*[
            nn.Conv2d(dim_in, num_heads, kernel_size, padding=padding_q, stride=stride_q), 
            nn.ReLU(), 
            nn.Conv2d(num_heads, num_heads, 1, groups=num_heads),
            nn.Softmax2d()
        ])

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                #('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        assert self.conv_proj_q is not None and not self.with_cls_token


        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        q = self.conv_proj_q(x)
        if VERB: print('        q    ', q.shape)

        q = rearrange(x, 'b (h d) x y -> b h d x y', h = self.num_heads)
        q = rearrange(q, 'b h d x y -> b h 1 d x y')
        if VERB: print('        q    ', q.shape)

        mask = self.mask_addon(x)
        if VERB: print('        mask', mask.shape)
        mask = rearrange(mask, 'b (h d) x y -> b h d x y', h = self.num_heads)
        mask = rearrange(mask, 'b h m x y -> b h m 1 x y')
        if VERB: print('        mask ', mask.shape)

        q_masked = q * mask
        q_masked = q_masked.sum(dim=(-1, -2))
        if VERB: print('        qmask', q_masked.shape)


        return q_masked, mask

    def forward(self, x, h, w):
        if VERB: print('   ', self.__class__.__name__)
        if VERB: print('      x h w', x.shape, h, w)

        q, mask = self.forward_conv(x, h, w)
        
        if VERB: print('      q', q.shape)
        if VERB: print('      pk', self.proto_k.shape)
        if VERB: print('      pv', self.proto_v.shape)

        #exit()

        k = e_repeat(self.proto_k, 'h m d -> b h m d', b=x.shape[0])
        v = e_repeat(self.proto_v, 'h m d -> b h m d', b=x.shape[0])
        if VERB: print('      *q', q.shape)
        if VERB: print('      *k', k.shape)
        if VERB: print('      *v', v.shape)
        
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        #attn = F.softmax(attn_score, dim=-1)
        attn = self.attend(attn_score)
        attn = self.attn_drop(attn)
        if VERB: print('      attn', attn.shape)

        #corr_v = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        #if VERB: print('    cv  ', corr_v.shape)
        #corr_v = rearrange(corr_v, 'b h m d -> b h m d')
        #print('    cv  ', corr_v.shape)

        if VERB: print('      mask', mask.shape)
        mask = rearrange(mask, 'b h m c x y -> b h (c x y) m')
        if VERB: print('      mask', mask.shape)

        mask_attn = torch.einsum('bhmm,bhtm->bhtm', [attn, mask])
        if VERB: print('      mska', mask_attn.shape)

        x = torch.einsum('bhpm,bhmd->bhpd', [mask_attn, v])
        if VERB: print('      x   ', x.shape)

        x = rearrange(x, 'b h t d -> b t (h d)')
        if VERB: print('      xr', x.shape)

        x = self.proj(x)
        if VERB: print('      xp', x.shape)
        #print('    -------------------')
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
            hasattr(module, 'conv_proj_q')
            and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
            hasattr(module, 'conv_proj_k')
            and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
            hasattr(module, 'conv_proj_v')
            and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 attn_layer=Attention,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = attn_layer(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, h, w):
        if VERB: print(' ', self.__class__.__name__)
        res = x
        if VERB: print('    x      ', x.shape)
        if VERB: print('    res    ', res.shape)
        x = self.norm1(x)
        if VERB: print('    x-n    ', x.shape)
        attn = self.attn(x, h, w)
        if VERB: print('    attn   ', attn.shape)
        x = res + self.drop_path(attn)
        if VERB: print('    r+attn ', x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if VERB: print('    rx+mlp ', x.shape)

        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 attn_layers=[],
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    attn_layer=eval(attn_layers[j]),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if VERB: print(self.__class__.__name__)
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        if VERB: print('  x pecon', x.shape)
        x = rearrange(x, 'b c h w -> b (h w) c')
        if VERB: print('  x rearr', x.shape)

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            if VERB: print('  x+token', x.shape)


        x = self.pos_drop(x)
        if VERB: print('  x> blk ', x.shape)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
        if VERB: print('  >x bout', x.shape)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H*W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        if VERB: print('  x rshp', x.shape)
        return x, cls_tokens


class R2RConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 #num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        #self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
                #'n_mask': spec['N_MASK'][i] if 'N_MASK' in spec else None
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                attn_layers=spec['ATTN_LAYER'][i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        #self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        #trunc_normal_(self.head.weight, std=0.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        if VERB: print(self.__class__.__name__)
        for i in range(self.num_stages):
            if VERB: print('stage', i)
            if VERB: print('x> in', x.shape)
            x, cls_tokens = getattr(self, f'stage{i}')(x)
            if VERB: print('>x out', x.shape)

        if self.cls_token:
            if VERB: print('cls_token', cls_tokens.shape)
            x = self.norm(cls_tokens)
            if VERB: print('x-cls n', x.shape)
            x = torch.squeeze(x)
            if VERB: print('x sqz', x.shape)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            if VERB: print('x rear', x.shape)
            x = self.norm(x)
            if VERB: print('x n', x.shape)
            x = torch.mean(x, dim=1)
            if VERB: print('x mean', x.shape)

        if VERB: print('x out', x.shape)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)

        return x




from networks.layer_builder import build_feat_extractor, build_layer
class Region2RegionNet(nn.Module):
    def __init__(self, num_classes, context_encoder, last_layer, backbone=None, proto_layer=None):
        super().__init__()
        if backbone is not None:
            self.backbone, last_in_feat, last_out_feat = build_feat_extractor(**backbone)
        else:
            self.backbone = None
            last_in_feat = 3

        self.context_encoder = build_context_encoder(**context_encoder)

        if proto_layer is not None:
            self.proto_layer = build_layer(in_channels=last_in_feat, num_classes=num_classes, **proto_layer)
        else:
            self.proto_layer = None

        self.last_layer = build_layer(num_classes=num_classes, **last_layer)


    def forward(self, x, **cls_args):
        # Extract feats
        if self.backbone is not None:
            feats = self.backbone(x)
        else:
            feats = x

        # Transformer (context encoder)
        xformer_feat = self.context_encoder(feats)

        # Apply prototype
        if self.proto_layer is not None:
            y, proto_info = self.proto_layer(xformer_feat, **cls_args)
        else:
            y, proto_info = xformer_feat, {}
        
        # Apply last layer
        y = self.last_layer(y)
        if type(y) == tuple:
            y, cls_info = y
        else:
            cls_info = {}

        model_info = {**proto_info, **cls_info}
        return y, model_info


    def warm_only(self):
        raise NotImplementedError()


    def joint(self):
        raise NotImplementedError()



def build_context_encoder(arch, pretrain=None, **kwargs):
    if arch == 'r2r_cvt':
        net = R2RConvolutionalVisionTransformer(**kwargs)
        if pretrain is not None:
            d = torch.load(pretrain)
            net.load_state_dict(d, strict=False)
    else:
        raise Exception('Unknown context encoder arch ' + str(arch))

    return net 


def test_attn():
    attn = Attention_v5(dim_in=192,
                 dim_out=192,
                 n_mask=17,
                 num_heads=3,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False)
    x = torch.ones(7, 16, 192)
    print('x', x.shape)
    y = attn(x, 4, 4)
    print('y', y.shape)


def test_cvt():
    net = R2RConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=14,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init='trunc_norm',
        spec={
            #'N_MASK':[56*56, 28*28, 14*14],
            'ATTN_LAYER': ['Attention_v5', 'Attention','Attention'],

            'INIT': 'trunc_norm',
            'NUM_STAGES': 3,
            'PATCH_SIZE': [7, 3, 3],
            'PATCH_STRIDE': [4, 2, 2],
            'PATCH_PADDING': [2, 1, 1],
            'DIM_EMBED': [64, 192, 384],
            'NUM_HEADS': [1, 3, 6],
            'DEPTH': [1, 2, 10],
            'MLP_RATIO': [4.0, 4.0, 4.0],
            'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
            'DROP_RATE': [0.0, 0.0, 0.0],
            'DROP_PATH_RATE': [0.0, 0.0, 0.1],
            'QKV_BIAS': [True, True, True],
            'CLS_TOKEN': [False, False, True],
            'POS_EMBED': [False, False, False],
            'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
            'KERNEL_QKV': [3, 3, 3],
            'PADDING_KV': [1, 1, 1],
            'STRIDE_KV': [2, 2, 2],
            'PADDING_Q': [1, 1, 1],
            'STRIDE_Q': [1, 1, 1]
        }
    )

    print(net)

    x = torch.ones(7, 3, 256, 256)
    print('x', x.shape)
    y = net(x)
    print('y', y.shape)


VERB = False
if __name__ == '__main__':
    #test_attn()
    test_cvt()