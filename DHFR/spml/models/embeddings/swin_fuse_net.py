import torch
import torch.nn as nn
import sys
import time
import numpy as np
import torchvision.models as models
from torch.nn import functional as F

# debug
'''
import sys
sys.path.append("/home/liuzhy/SaliencyTrain/SFWSOD")
from sa_transformer_block_debug import SwinSaBlock
from rgbd_swin_encoder_debug import rgbd_swin_encoder_tiny_patch4_window7_224, RgbDMerging
from localvit_swin_encoder_debug import localvit_swin_encoder_tiny_patch4_window7_224
'''
from spml.models.embeddings.localvit_swin_encoder import localvit_swin_encoder_tiny_patch4_window7_224
from spml.models.embeddings.swin_fuse_block import swin_fuse_block, CrossPaddingAttentionBlock

class UpsampleHead(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(UpsampleHead, self).__init__()
        self.conv1 = self.conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        
    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + x
        out = nn.functional.upsample(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv2(out)        
        out = self.bn2(out)
        out = self.relu(out)
        return out

#SwinFusingNet
class SFNet(nn.Module):
    def __init__(self, need_depth_estimate=False):
        super(SFNet, self).__init__()
        
        # config
        self.use_sal_mask = True
        
        # Backbone model
        if self.training:
          self.rgb_swin = localvit_swin_encoder_tiny_patch4_window7_224(pretrain=True)
          self.depth_swin = localvit_swin_encoder_tiny_patch4_window7_224(pretrain=True)
          self.fuse_swin = swin_fuse_block(pretrain=True, use_sal_mask=self.use_sal_mask)
        else:
          self.rgb_swin = localvit_swin_encoder_tiny_patch4_window7_224(pretrain=False)
          self.depth_swin = localvit_swin_encoder_tiny_patch4_window7_224(pretrain=False)
          self.fuse_swin = swin_fuse_block(pretrain=False, use_sal_mask=self.use_sal_mask)
        
        # translate layers
        self.enc_div16_trans = nn.Conv2d(384 * 2, 256, 1)
        self.enc_div8_trans = nn.Conv2d(192 * 2, 128, 1)
        self.enc_div4_trans = nn.Conv2d(96 * 2, 64, 1)
        self.div16_trans = nn.Conv2d(384, 256, 1)
        
        # decoder fuse to div8
        self.dec_div8_merging = CrossPaddingAttentionBlock(
            output_resolution=(28, 28), dim=128, depth=4, num_heads=8, in_is_tokens=False, use_sal_mask=self.use_sal_mask)

        # decoder fuse to div4
        self.dec_div4_merging = CrossPaddingAttentionBlock(
            output_resolution=(56, 56), dim=64, depth=4, num_heads=8, in_is_tokens=False, use_sal_mask=self.use_sal_mask)

        # decoder fuse to div2
        self.dec_div2_merging = CrossPaddingAttentionBlock(
            output_resolution=(112, 112), dim=32, depth=0, num_heads=8, in_is_tokens=False, use_sal_mask=False)
        self.upsample_head = UpsampleHead(32, 32)
        self.contour_upsample_head1 = UpsampleHead(64, 32)
        self.contour_upsample_head2 = UpsampleHead(32, 32)
        
        # classify
        self.classifier_before_fuse_div16 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=True)
        self.classifier_contour_div1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=True)
        self.classifier_div16 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=True)
        self.classifier_div8 = nn.Conv2d(128, 1, kernel_size=1, stride=1, bias=True)
        self.classifier_div4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=True)
        self.classifier_div2 = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=True)
        self.classifier_div1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=True)

        # depth prediction branch
        self.need_depth_estimate = need_depth_estimate
        if self.need_depth_estimate:
            self.depth_trans_div4 = nn.Conv2d(96, 64, 1)
            self.depth_trans_div8 = nn.Conv2d(192, 64, 1)
            self.depth_trans_div16 = nn.Conv2d(384, 64, 1)
            self.depth_div16_to_div16 = self.conv_block(64, 64)
            self.depth_fuse_div16_to_div8 = self.conv_block(128, 64)
            self.depth_fuse_div8_to_div4 = self.conv_block(128, 64)
            self.depth_classifier = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=True)

    def conv_block(self, inplanes, planes, kernel_size=3, padding=1):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                      stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def forward(self, x, x_depth):
        B, C, H, W = x.shape
        
        # encoder
        rgb_enc_div4, rgb_enc_div8, rgb_enc_div16 = self.rgb_swin(x)
        d_enc_div4, d_enc_div8, d_enc_div16 = self.depth_swin(x_depth)
        enc_div16 = self.enc_div16_trans(torch.cat((rgb_enc_div16, d_enc_div16), dim=1))
        enc_div8 = self.enc_div8_trans(torch.cat((rgb_enc_div8, d_enc_div8), dim=1))
        enc_div4 = self.enc_div4_trans(torch.cat((rgb_enc_div4, d_enc_div4), dim=1))
        before_fuse_div16 = self.classifier_before_fuse_div16(enc_div16)
        before_fuse_div16 = torch.sigmoid(
            F.interpolate(before_fuse_div16, size=(H, W), mode='bilinear', align_corners=False))
        del rgb_enc_div4, rgb_enc_div8

        # swin fuse
        div16_sal_mask = F.interpolate(before_fuse_div16, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        x_dict = {'feat1': rgb_enc_div16, 'feat2': d_enc_div16, 'sal_mask': div16_sal_mask}
        swin_fuse = self.fuse_swin(x_dict)['feat']
        dec_div16 = self.div16_trans(swin_fuse)
        div16 = self.classifier_div16(dec_div16)
        div16 = torch.sigmoid(F.interpolate(div16, size=(H, W), mode='bilinear', align_corners=False))
        del rgb_enc_div16, swin_fuse
        
        # decoder div16 merge
        div8_sal_mask = F.interpolate(div16, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        x_dict = {'feat1': enc_div16, 'feat2': dec_div16, 'sal_mask': div8_sal_mask}
        dec_div8 = self.dec_div8_merging(x_dict)['feat']
        div8 = self.classifier_div8(dec_div8)
        div8 = torch.sigmoid(F.interpolate(div8, size=(H, W), mode='bilinear', align_corners=False))
        del enc_div16
        
        # decoder div8 merge
        div4_sal_mask = F.interpolate(div8, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        x_dict = {'feat1': enc_div8, 'feat2': dec_div8, 'sal_mask': div4_sal_mask}
        dec_div4 = self.dec_div4_merging(x_dict)['feat']

        div4 = self.classifier_div4(dec_div4)
        div4 = torch.sigmoid(F.interpolate(div4, size=(H, W), mode='bilinear', align_corners=False))
        del enc_div8
        
        # decoder div2 merge
        x_dict = {'feat1': enc_div4, 'feat2': dec_div4}
        dec_div2 = self.dec_div2_merging(x_dict)['feat']
        div2 = self.classifier_div2(dec_div2)
        div2 = torch.sigmoid(F.interpolate(div2, size=(H, W), mode='bilinear', align_corners=False))
        dec_div1 = self.upsample_head(dec_div2)
        del enc_div4, dec_div2

        div1 = torch.sigmoid(self.classifier_div1(dec_div1))
		
		# depth prediction
        if self.need_depth_estimate:
            # trans
            d_enc_div4 = self.depth_trans_div4(d_enc_div4)
            d_enc_div8 = self.depth_trans_div8(d_enc_div8)
            d_enc_div16 = self.depth_trans_div16(d_enc_div16)

            # dec16
            d_dec_div16 = self.depth_div16_to_div16(d_enc_div16)

            # dec8
            d_dec_div16 = F.interpolate(d_dec_div16, scale_factor=2, mode='bilinear', align_corners=True)
            d_dec_div8 = self.depth_fuse_div16_to_div8(torch.cat((d_dec_div16, d_enc_div8), dim=1))

            # dec4
            d_dec_div8 = F.interpolate(d_dec_div8, scale_factor=2, mode='bilinear', align_corners=True)
            d_dec_div4 = self.depth_fuse_div8_to_div4(torch.cat((d_dec_div8, d_enc_div4), dim=1))

            # upsample
            d_dec_div1 = F.interpolate(d_dec_div4, scale_factor=4, mode='bilinear', align_corners=True)

            # classify
            d_dec_div1 = self.depth_classifier(d_dec_div1).sigmoid()

        # classify
        contour_div2 = self.contour_upsample_head1(dec_div4)
        contour_div1 = self.contour_upsample_head2(contour_div2)
        contour_div1 = self.classifier_contour_div1(contour_div1)
        contour_div1 = torch.sigmoid(F.interpolate(contour_div1, size=(H, W), mode='bilinear', align_corners=False))

        return before_fuse_div16, div16, div8, div4, div2, div1, dec_div4, contour_div1


if __name__=="__main__":
    rgb = torch.rand(6, 3, 224, 224).cuda()
    d = torch.rand(6, 3, 224, 224).cuda()
    model = SFNet().cuda()
    model.train()
    startT = time.time()
    for i in range(100):
        out = model(rgb, d)
        del out
    endT = time.time()
    print('time=', endT - startT)
