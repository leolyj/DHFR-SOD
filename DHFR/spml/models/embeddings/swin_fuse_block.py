"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch

Introducing locality mechanism to "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
'''
# debug
from swin_transformer import window_partition, window_reverse, WindowAttention, PatchMerging, PatchEmbed
from swin_transformer import SwinTransformer
from localvit import LocalityFeedForward
'''

import numpy as np
from spml.models.embeddings.masked_wsa_block import MaskedWSABlock, window_partition, window_reverse



class CrossPadding(nn.Module):
  def __init__(self, output_resolution, dim, pos_drop=0., norm_layer=nn.LayerNorm, in_is_tokens=True):
    super().__init__()
    # depth-wise conv
    #self.depth_cnn = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
    #self.point_cnn = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1)

    # other params
    self.output_resolution = output_resolution
    self.dim = dim
    self.norm = norm_layer(dim)
    self.in_is_tokens = in_is_tokens
    self.pos_drop = nn.Dropout(p=pos_drop)

  def forward(self, x_dict):
    feat1 = x_dict['feat1']
    feat2 = x_dict['feat2']

    H, W = self.output_resolution
    if self.in_is_tokens:
      B, Lr, Cr = feat1.shape
      B, Ld, Cd = feat2.shape
      Wr = Hr = np.round(np.sqrt(Lr))
      Wd = Hd = np.round(np.sqrt(Ld))
      feat1 = feat1.transpose(-2, -1).view(B, Cr, Hr, Wr)
      feat2 = feat2.transpose(-2, -1).view(B, Cd, Hd, Wd)
    else:
      B, Cr, Hr, Wr = feat1.shape
      B, Cd, Hd, Wd = feat2.shape
    assert Hr == Hd and Hr == (H // 2) and Wr == Wd and Wr == (W // 2) and Cr == Cd and (
          Cr // 2) == self.dim, "input feature has wrong size"

    # padding, feat1 afford 0 3, feat2 afford 1 2
    '''
    if self.training:
      idxList = np.arange(4)
      np.random.shuffle(idxList)
      idxMap = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
      idxMap = idxMap[idxList]
      mergeFeat = feat1[:, 0::2, :, :].repeat(1, 1, 2, 2)
      mergeFeat[:, :, idxMap[0][0]::2, idxMap[0][1]::2] = feat1[:, 0::2, :, :] # pos 0
      mergeFeat[:, :, idxMap[1][0]::2, idxMap[1][1]::2] = feat2[:, 0::2, :, :] # pos 1
      mergeFeat[:, :, idxMap[2][0]::2, idxMap[2][1]::2] = feat2[:, 1::2, :, :] # pos 2
      mergeFeat[:, :, idxMap[3][0]::2, idxMap[3][1]::2] = feat1[:, 1::2, :, :] # pos 3
    else:
      mergeFeat = feat1[:, 0::2, :, :].repeat(1, 1, 2, 2)
      mergeFeat[:, :, 0::2, 0::2] = feat1[:, 0::2, :, :]
      mergeFeat[:, :, 0::2, 1::2] = feat2[:, 0::2, :, :]
      mergeFeat[:, :, 1::2, 0::2] = feat2[:, 1::2, :, :]
      mergeFeat[:, :, 1::2, 1::2] = feat1[:, 1::2, :, :]
    '''
    mergeFeat = feat1[:, 0::2, :, :].repeat(1, 1, 2, 2)
    mergeFeat[:, :, 0::2, 0::2] = feat1[:, 0::2, :, :]
    mergeFeat[:, :, 0::2, 1::2] = feat2[:, 0::2, :, :]
    mergeFeat[:, :, 1::2, 0::2] = feat2[:, 1::2, :, :]
    mergeFeat[:, :, 1::2, 1::2] = feat1[:, 1::2, :, :]
    mergeFeat = mergeFeat.view(B, self.dim, H * W).transpose(-2, -1)  # B H*W C/2

    # norm
    mergeFeat = self.norm(mergeFeat)
    
    # drop
    mergeFeat = self.pos_drop(mergeFeat)

    # low filter
    mergeFeat = mergeFeat.transpose(-2, -1).view(B, self.dim, H, W)
    #mergeFeat = self.point_cnn(self.depth_cnn(mergeFeat))

    if self.in_is_tokens:
      mergeFeat = mergeFeat.view(B, self.dim, -1).transpose(-2, -1)

    x_dict['feat'] = mergeFeat

    return x_dict


class CrossPaddingAttentionBlock(nn.Module):
    def __init__(self, output_resolution=(28, 28), dim=192, depth=2, num_heads=6,
                 window_size=7, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 in_is_tokens=True, use_sal_mask=False):
        super().__init__()
        # cross padding
        self.cross_padding = CrossPadding(output_resolution, dim, pos_drop=drop_rate, norm_layer=norm_layer, in_is_tokens=in_is_tokens)

        # window attention layers
        self.depth = depth
        if self.depth > 0:
          self.layers = nn.ModuleList()
          for i_layer in range(depth):
            self.layers.append(
              MaskedWSABlock(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                             attn_drop=attn_drop_rate, drop_path=drop_path_rate/11*(2+i_layer), norm_layer=norm_layer,
                             n_pos=output_resolution[0] * output_resolution[1], in_is_tokens=in_is_tokens,
                             window_size=window_size, shift_size=0 if (i_layer % 2 == 0) else window_size // 2,
                             use_sal_mask=use_sal_mask))

    def forward(self, x_dict):
        x_dict = self.cross_padding(x_dict)
        if self.depth > 0:
          for layer in self.layers:
            x_dict = layer(x_dict)
        return x_dict


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, in_is_tokens=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.in_is_tokens = in_is_tokens

    def forward(self, x_dict):
        x = x_dict['feat']

        H, W = self.input_resolution
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        if not self.in_is_tokens:
          B, C, Hx, Wx = x.shape
          x = x.permute(0, 2, 3, 1).contiguous()
          assert Hx == H and Wx == W, "input feature has wrong size"
        else:
          B, L, C = x.shape
          assert L == H * W, "input feature has wrong size"
          x = x.view(B, H, W, C)

        # padding, feat1 afford 0 3, feat2 afford 1 2
        '''
        if self.training:
          idxList = np.arange(4)
          np.random.shuffle(idxList)
          idxMap = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
          idxMap = idxMap[idxList]
          x0 = x[:, idxMap[0][0]::2, idxMap[0][1]::2, :]  # B H/2 W/2 C
          x1 = x[:, idxMap[1][0]::2, idxMap[1][1]::2, :]  # B H/2 W/2 C
          x2 = x[:, idxMap[2][0]::2, idxMap[2][1]::2, :]  # B H/2 W/2 C
          x3 = x[:, idxMap[3][0]::2, idxMap[3][1]::2, :]  # B H/2 W/2 C
        else:
          x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
          x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
          x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
          x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        '''
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        if not self.in_is_tokens:
            x = x.transpose(-2, -1).view(B, C * 2, H // 2, W // 2)

        x_dict['feat'] = x
        return x_dict


class PatchMergingAttentionBlock(nn.Module):
  def __init__(self, input_resolution=(28, 28), dim=192, depth=2, num_heads=12,
               window_size=7, qkv_bias=True, qk_scale=None, drop_rate=0.,
               attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
               in_is_tokens=True, use_sal_mask=False):
    super().__init__()
    # patch merging
    self.patch_merging = PatchMerging(input_resolution, dim, norm_layer=norm_layer, in_is_tokens=in_is_tokens)

    # window attention layers
    self.depth = depth
    self.output_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
    self.output_dim = dim * 2
    if self.depth > 0:
      self.layers = nn.ModuleList()
      for i_layer in range(depth):
        self.layers.append(
          MaskedWSABlock(self.output_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=drop_path_rate/11*(4+i_layer), norm_layer=norm_layer,
                         n_pos=self.output_resolution[0] * self.output_resolution[1], in_is_tokens=in_is_tokens,
                         window_size=window_size, shift_size=0 if (i_layer % 2 == 0) else window_size // 2,
                         use_sal_mask=use_sal_mask))

  def forward(self, x_dict):
    x_dict = self.patch_merging(x_dict)
    if self.depth > 0:
      if 'sal_mask' in x_dict.keys():
        if x_dict['sal_mask'] is not None:
          x_dict['sal_mask'] = F.interpolate(x_dict['sal_mask'], size=self.output_resolution, mode='bilinear', align_corners=False)
      for layer in self.layers:
        x_dict = layer(x_dict)
    return x_dict


class SwinFuseBlock(nn.Module):

    def __init__(self, input_resolution=(14, 14), dim=384, depths=[2, 2], num_heads=[6, 12],
               window_size=7, qkv_bias=True, qk_scale=None, drop_rate=0.,
               attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
               in_is_tokens=False, use_sal_mask=False):
      super().__init__()

      # Cross Padding Attention Block
      output_resolution = (input_resolution[0] * 2, input_resolution[1] * 2)
      output_dim = dim // 2
      self.cpab = CrossPaddingAttentionBlock(output_resolution, dim=output_dim, depth=depths[0], num_heads=num_heads[0],
                                             window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                             drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                             in_is_tokens=in_is_tokens, use_sal_mask=use_sal_mask)

      # Patch Merging Attention Block
      self.pmab = PatchMergingAttentionBlock(output_resolution, dim=output_dim, depth=depths[1], num_heads=num_heads[1],
                                             window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                             drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                             in_is_tokens=in_is_tokens, use_sal_mask=use_sal_mask)

      self.apply(self._init_weights)

    def _init_weights(self, m):
      if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
      return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
      return {'relative_position_bias_table'}

    def forward(self, x_dict):
        x_dict = self.cpab(x_dict)
        x_dict = self.pmab(x_dict)
        return x_dict

@register_model
def swin_fuse_block(pretrain=True, use_sal_mask=True, **kwargs):
    model = SwinFuseBlock(use_sal_mask=use_sal_mask)

    if pretrain:
        checkpoint = torch.load('./snapshots/localvit/localvit_swin.pth', map_location='cpu')
        checkpoint_model = checkpoint['model']
        
        model_params = model.state_dict()

        # init
        for key in model_params.keys():
          newLayerKey = None
          if 'cpab.layers.0.w_attn' in key:
            newLayerKey = key.replace('cpab.layers.0.w_attn', 'layers.1.blocks.0')
          if 'cpab.layers.1.w_attn' in key:
            newLayerKey = key.replace('cpab.layers.1.w_attn', 'layers.1.blocks.1')
          if 'pmab.patch_merging' in key:
            newLayerKey = key.replace('pmab.patch_merging', 'layers.1.downsample')
          if 'pmab.layers.0.w_attn' in key:
            newLayerKey = key.replace('pmab.layers.0.w_attn', 'layers.2.blocks.0')
          if 'pmab.layers.1.w_attn' in key:
            newLayerKey = key.replace('pmab.layers.1.w_attn', 'layers.2.blocks.1')
          if 'pmab.layers.2.w_attn' in key:
            newLayerKey = key.replace('pmab.layers.2.w_attn', 'layers.2.blocks.2')
          if 'pmab.layers.3.w_attn' in key:
            newLayerKey = key.replace('pmab.layers.3.w_attn', 'layers.2.blocks.3')
          if 'pmab.layers.4.w_attn' in key:
            newLayerKey = key.replace('pmab.layers.4.w_attn', 'layers.2.blocks.4')
          if 'pmab.layers.5.w_attn' in key:
            newLayerKey = key.replace('pmab.layers.5.w_attn', 'layers.2.blocks.5')
          if newLayerKey is not None:
            model_params[key] = checkpoint_model[newLayerKey]

        model.load_state_dict(model_params, strict=True)
        print('load swin_fuse_block pretrain model successful!')

    return model


if __name__=="__main__":
  model = swin_fuse_block()


