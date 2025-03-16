# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
import math
import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from spml.models.embeddings.localvit import LocalityFeedForward


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_sal_mask=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.use_sal_mask = use_sal_mask
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, sal_fg_attn_mask=None, sal_bg_attn_mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if self.use_sal_mask:
            attn_fg = attn.view(-1, self.num_heads, N, N) + sal_fg_attn_mask.unsqueeze(1)
            attn_bg = attn.view(-1, self.num_heads, N, N) + sal_bg_attn_mask.unsqueeze(1)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            if self.use_sal_mask:
                attn_fg = attn_fg.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn_fg = attn_fg.view(-1, self.num_heads, N, N)
                attn_fg = self.softmax(attn_fg)
                attn_bg = attn_bg.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn_bg = attn_bg.view(-1, self.num_heads, N, N)
                attn_bg = self.softmax(attn_bg)
        else:
            attn = self.softmax(attn)
            if self.use_sal_mask:
                attn_fg = self.softmax(attn_fg)
                attn_bg = self.softmax(attn_bg)

        attn = self.attn_drop(attn)
        if self.use_sal_mask:
            attn_fg = self.attn_drop(attn_fg)
            attn_bg = self.attn_drop(attn_bg)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.use_sal_mask:
            x_fg = (attn_fg @ v).transpose(1, 2).reshape(B_, N, C)
            x_bg = (attn_bg @ v).transpose(1, 2).reshape(B_, N, C)
            x = x + x_fg - x_bg
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class MaskedWSABlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_depthwise=True,
                 use_pos=False, n_pos=None, in_is_tokens=True, window_size=7, shift_size=0,
                 use_sal_mask=False):
        super().__init__()
        self.w_attn = MaskedWSABaseBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                      act_layer=act_layer, norm_layer=norm_layer, use_depthwise=use_depthwise,
                                      use_pos=use_pos, n_pos=n_pos, in_is_tokens=in_is_tokens, window_size=window_size,
                                      shift_size=shift_size, use_sal_mask=use_sal_mask)
    def forward(self, x_dict):
        sal_mask = None
        if 'sal_mask' in x_dict.keys():
            sal_mask = x_dict['sal_mask']
        x = x_dict['feat']
        x = self.w_attn(x, sal_mask)
        x_dict['feat'] = x
        return x_dict


class MaskedWSABaseBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_depthwise=True,
                 use_pos=False, n_pos=None, in_is_tokens=True, window_size=7, shift_size=0,
                 use_sal_mask=False):
        super().__init__()
        self.in_is_tokens = in_is_tokens
        self.use_pos = use_pos
        if self.use_pos:
            self.pos_emb = nn.Parameter(get_sinusoid_encoding(n_pos, dim), requires_grad=False)
        self.dim = dim
        self.input_resolution = (int(math.sqrt(n_pos)), int(math.sqrt(n_pos)))
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_depthwise = use_depthwise
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_sal_mask=use_sal_mask)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if not use_depthwise:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, reduction=dim // 4)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, sal_mask=None):
        if not self.in_is_tokens:
            Bx, Cx, Hx, Wx = x.shape
            x = x.view(Bx, Cx, -1).transpose(-2, -1)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        if self.use_pos:
            x = x + self.pos_emb
            x = self.drop_path(x)
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        sal_fg_attn_mask = None
        sal_bg_attn_mask = None
        if sal_mask is not None:
            fg_mask_windows = window_partition((sal_mask >= 0.8).permute(0, 2, 3, 1).contiguous(),
                                               self.window_size)  # nW, window_size, window_size, 1
            fg_mask_windows = fg_mask_windows.view(-1, self.window_size * self.window_size).int()
            fg_mask_windows_t = fg_mask_windows.clone()
            fg_mask_windows_t[fg_mask_windows == 0] = 2 # make sure only attention on forground
            sal_fg_attn_mask = fg_mask_windows.unsqueeze(1) - fg_mask_windows_t.unsqueeze(2)
            sal_fg_attn_mask = sal_fg_attn_mask.masked_fill(sal_fg_attn_mask != 0, float(-100.0)).masked_fill(
                sal_fg_attn_mask == 0,
                float(0.0))

            bg_mask_windows = window_partition((sal_mask < 0.2).permute(0, 2, 3, 1).contiguous(),
                                               self.window_size)  # nW, window_size, window_size, 1
            bg_mask_windows = bg_mask_windows.view(-1, self.window_size * self.window_size).int()
            bg_mask_windows_t = bg_mask_windows.clone()
            bg_mask_windows_t[bg_mask_windows == 0] = 2 # make sure only attention on background
            sal_bg_attn_mask = bg_mask_windows.unsqueeze(1) - bg_mask_windows_t.unsqueeze(2)
            sal_bg_attn_mask = sal_bg_attn_mask.masked_fill(sal_bg_attn_mask != 0, float(-100.0)).masked_fill(
                sal_bg_attn_mask == 0,
                float(0.0))
        attn_windows = self.attn(x_windows, mask=self.attn_mask, sal_fg_attn_mask=sal_fg_attn_mask, sal_bg_attn_mask=sal_bg_attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        if not self.use_depthwise:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.conv(x.view(B, H, W, C).permute(0, 3, 1, 2))
            x = x.permute(0, 2, 3, 1).view(B, H * W, C)

        if not self.in_is_tokens:
            x = x.transpose(-2, -1).view(B, C, H, W)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


from thop import profile, clever_format

if __name__=="__main__":
    x = torch.rand(4, 56*56, 96)
    saTest = SwinSaBlock(96, use_pos=True, n_pos=56*56)
    startT = time.time()
    out = saTest(x)
    endT = time.time()
    print('time=', endT-startT)
    print(out.shape)

    macs, params = profile(saTest, inputs=(x, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('macs=', macs, ',params=', params)
