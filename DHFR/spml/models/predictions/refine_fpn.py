import torch
import torch.nn as nn
import torchvision.models as models

# debug
'''
import sys
sys.path.append("/home/liuzhy/SaliencyTrain/SPML")
sys.path.append("/home/liuzhy/SaliencyTrain/SFWSOD_swin/spml/models/backbones")
from ResNet50 import ResNet50
'''

from spml.models.backbones.ResNet50 import ResNet50
from torch.nn import functional as F
import os
from PIL import Image


#RefineFPN
class RefineFPN(nn.Module):
    def __init__(self):
        super(RefineFPN, self).__init__()
        
        #Backbone model
        self.resnet = ResNet50('rgbd')

        #upsample function
        self.upsampleX2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #fuse
        self.trans_div2 = nn.Conv2d(64, 64, 1)
        self.trans_div4 = nn.Conv2d(256, 64, 1)
        self.trans_div8 = nn.Conv2d(512, 64, 1)
        self.trans_div16 = nn.Conv2d(1024, 64, 1)
        self.div16_to_div16 = self._make_agant_layer3x3(64, 64)
        self.fuse_div16_to_div8 = self._make_agant_layer3x3(128, 64)
        self.fuse_div8_to_div4 = self._make_agant_layer3x3(128, 64)
        self.fuse_div4_to_div2 = self._make_agant_layer3x3(128, 64)
        self.classifier = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=True)

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        # div2
        enc_div2 = self.trans_div2(x)
        x = self.resnet.maxpool(x)

        # div4
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        enc_div4 = self.trans_div4(x1)
        del x

        # div8
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        enc_div8 = self.trans_div8(x2)
        del x1

        # div16
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        enc_div16 = self.trans_div16(x3)
        del x2, x3

        # decoder16
        dec_div16 = self.div16_to_div16(enc_div16)
        del enc_div16

        # div16 div8 merge
        dec_div8 = self.upsampleX2(dec_div16)
        dec_div8 = self.fuse_div16_to_div8(torch.cat((enc_div8, dec_div8), dim=1))
        del enc_div8

        # div8 div4 merge
        dec_div4 = self.upsampleX2(dec_div8)
        dec_div4 = self.fuse_div8_to_div4(torch.cat((enc_div4, dec_div4), dim=1))
        del enc_div4
        
        # div4 div2 merge
        dec_div2 = self.upsampleX2(dec_div4)
        dec_div2 = self.fuse_div4_to_div2(torch.cat((enc_div2, dec_div2), dim=1))
        del enc_div2
        
        # classify
        dec_div1 = self.upsampleX2(dec_div2)
        dec_div1 = self.classifier(dec_div1).sigmoid()
        
        return dec_div1
        
    def _make_agant_layer3x3(self, inplanes, planes, kernel_size=3, padding=1):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                      stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    #initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k=='conv1.weight':
                all_params[k]=torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


if __name__=="__main__":
    sal = torch.rand(1, 1, 224, 224).cuda()
    model = RefineFPN().cuda()
    startT = time.time()
    for i in range(0, 200):
        out = model(sal)
        del out
    endT = time.time()
    print('time=', endT - startT)
    print(out.shape)
