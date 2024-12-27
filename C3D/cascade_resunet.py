"""
ResUNet整体上网络结构基于VNet，做出的修改如下：
将原本555的卷积核换成了333
在除了第一个和最后一个block中添加了dropout
去掉了编码器部分的最后一个16倍降采样的stage
为了弥补这么做带来的感受野的损失，在编码器的最后两个stage加入了混合空洞卷积
"""

import torch
import torch.nn as nn

dropout_rate = 0.3

# 定义单个3D FCN
# class ResUNet(nn.Module):
#     """
#     共9332094个可训练的参数, 九百三十万左右
#     """
#
#     def __init__(self, training, inchannel, stage):
#         """
#         :param training: 标志网络是属于训练阶段还是测试阶段
#         :param inchannel 网络最开始的输入通道数量
#         :param stage 标志网络属于第一阶段，还是第二阶段
#         """
#         super().__init__()
#
#         self.training = training
#         self.stage = stage
#
#         self.encoder_stage1 = nn.Sequential(
#             nn.Conv3d(inchannel, 16, 3, 1, padding=1),
#             nn.PReLU(16),
#         )
#
#         self.encoder_stage2 = nn.Sequential(
#             nn.Conv3d(32, 32, 3, 1, padding=1),
#             nn.PReLU(32),
#
#             nn.Conv3d(32, 32, 3, 1, padding=1),
#             nn.PReLU(32),
#         )
#
#         self.encoder_stage3 = nn.Sequential(
#             nn.Conv3d(64, 64, 3, 1, padding=1),
#             nn.PReLU(64),
#
#             nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
#             nn.PReLU(64),
#
#             nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
#             nn.PReLU(64),
#         )
#
#         self.encoder_stage4 = nn.Sequential(
#             nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
#             nn.PReLU(128),
#
#             nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
#             nn.PReLU(128),
#
#             nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
#             nn.PReLU(128),
#         )
#
#         self.decoder_stage1 = nn.Sequential(
#             nn.Conv3d(128, 256, 3, 1, padding=1),
#             nn.PReLU(256),
#
#             nn.Conv3d(256, 256, 3, 1, padding=1),
#             nn.PReLU(256),
#
#             nn.Conv3d(256, 256, 3, 1, padding=1),
#             nn.PReLU(256),
#         )
#
#         self.decoder_stage2 = nn.Sequential(
#             nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
#             nn.PReLU(128),
#
#             nn.Conv3d(128, 128, 3, 1, padding=1),
#             nn.PReLU(128),
#
#             nn.Conv3d(128, 128, 3, 1, padding=1),
#             nn.PReLU(128),
#         )
#
#         self.decoder_stage3 = nn.Sequential(
#             nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
#             nn.PReLU(64),
#
#             nn.Conv3d(64, 64, 3, 1, padding=1),
#             nn.PReLU(64),
#         )
#
#         self.decoder_stage4 = nn.Sequential(
#             nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
#             nn.PReLU(32),
#         )
#
#         self.down_conv1 = nn.Sequential(
#             nn.Conv3d(16, 32, 2, 2),
#             nn.PReLU(32)
#         )
#
#         self.down_conv2 = nn.Sequential(
#             nn.Conv3d(32, 64, 2, 2),
#             nn.PReLU(64)
#         )
#
#         self.down_conv3 = nn.Sequential(
#             nn.Conv3d(64, 128, 2, 2),
#             nn.PReLU(128)
#         )
#
#         self.down_conv4 = nn.Sequential(
#             nn.Conv3d(128, 256, 3, 1, padding=1),
#             nn.PReLU(256)
#         )
#
#         self.up_conv2 = nn.Sequential(
#             nn.ConvTranspose3d(256, 128, 2, 2),
#             nn.PReLU(128)
#         )
#
#         self.up_conv3 = nn.Sequential(
#             nn.ConvTranspose3d(128, 64, 2, 2),
#             nn.PReLU(64)
#         )
#
#         self.up_conv4 = nn.Sequential(
#             nn.ConvTranspose3d(64, 32, 2, 2),
#             nn.PReLU(32)
#         )
#
#         self.map = nn.Conv3d(32, 1, 1)
#
#     def forward(self, inputs):
#         long_range1 = self.encoder_stage1(inputs)
#
#         short_range1 = self.down_conv1(long_range1)
#
#         long_range2 = self.encoder_stage2(short_range1) + short_range1
#         long_range2 = F.dropout(long_range2, dropout_rate, self.training)
#
#         short_range2 = self.down_conv2(long_range2)
#
#         long_range3 = self.encoder_stage3(short_range2) + short_range2
#         long_range3 = F.dropout(long_range3, dropout_rate, self.training)
#
#         short_range3 = self.down_conv3(long_range3)
#
#         long_range4 = self.encoder_stage4(short_range3) + short_range3
#         long_range4 = F.dropout(long_range4, dropout_rate, self.training)
#
#         short_range4 = self.down_conv4(long_range4)
#
#         outputs = self.decoder_stage1(long_range4) + short_range4
#         outputs = F.dropout(outputs, dropout_rate, self.training)
#
#         short_range6 = self.up_conv2(outputs)
#
#         outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
#         outputs = F.dropout(outputs, dropout_rate, self.training)
#
#         short_range7 = self.up_conv3(outputs)
#
#         outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
#         outputs = F.dropout(outputs, dropout_rate, self.training)
#
#         short_range8 = self.up_conv4(outputs)
#
#         outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
#
#         outputs = self.map(outputs)
#
#         # 返回概率图
#         return outputs

from resunet import ResUNet


class CascadeResUNet(nn.Module):
    def __init__(self, in_channel=3, num_filters=16):
        super(CascadeResUNet, self).__init__()

        self.stage1 = ResUNet(in_channel=in_channel, num_filters=num_filters, is_stage1=True)
        self.stage2 = ResUNet(in_channel=num_filters + in_channel, num_filters=num_filters)

        self.initialize()

    def forward(self, x):
        up, out_net_A = self.stage1(x)
        # print(up.shape, out_net_A.shape)
        out_net_B = self.stage2(torch.cat((up, x), dim=1))

        return [out_net_A, out_net_B]

    def initialize(self):
        # print('random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.stage1.modules)
        self.init_conv_IN(self.stage2.modules)

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)


if __name__ == '__main__':
    from torchsummary import summary

    in_channel = 3
    model = CascadeResUNet(in_channel=in_channel).to("cuda")

    summary(model, (in_channel, 128, 128, 128))
