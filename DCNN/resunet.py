from typing import Dict

import torch
import torch.nn as nn
from torchvision import models

from DCNN.efficientnet_unet import OutConv, Conv, UpConv


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDecoderBlock, self).__init__()
        middle_channels = int(in_channels // 2)

        self.conv1 = Conv(in_channels, middle_channels, kernel_size=3, dilation=1)
        self.up = UpConv(middle_channels, middle_channels)
        self.conv2 = Conv(middle_channels, out_channels, kernel_size=3, dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)

        return x


class ResUNet(nn.Module):
    def __init__(self, in_chans=4, num_classes=1, pretrain_backbone=True):
        super(ResUNet, self).__init__()
        self.stage_out_channels = [64, 64, 128, 256, 512]
        backbone = models.resnet34(pretrained=pretrain_backbone)
        self.firstconv = nn.Conv2d(in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # backbone = timm.create_model('resnet34', pretrained=pretrain_backbone, in_chans=in_chans)
        # print(backbone)
        # self.firstconv = backbone.conv1
        self.firstbn = backbone.bn1
        self.firstrelu = backbone.relu
        self.firstmaxpool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.up1 = ResDecoderBlock(self.stage_out_channels[4], self.stage_out_channels[3])
        self.up2 = ResDecoderBlock(self.stage_out_channels[3], self.stage_out_channels[2])
        self.up3 = ResDecoderBlock(self.stage_out_channels[2], self.stage_out_channels[1])
        self.up4 = ResDecoderBlock(self.stage_out_channels[1], self.stage_out_channels[0])
        self.outconv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0 = self.firstmaxpool(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # decoder
        d4 = self.up1(e4) + e3
        d3 = self.up2(d4) + e2
        d2 = self.up3(d3) + e1
        d1 = self.up4(d2)
        out = self.outconv(d1)

        return out


if __name__ == '__main__':
    from torchsummary import summary

    model = ResUNet(in_chans=4, num_classes=1, pretrain_backbone=True).to("cuda")
    summary(model, (4, 128, 128))
