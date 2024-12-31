import torch
import torch.nn as nn

from C3D.resunet import ResUNet


class CascadeResUNet(nn.Module):
    def __init__(self, in_channel=3, num_filters=16):
        super(CascadeResUNet, self).__init__()

        self.stage1 = ResUNet(in_channel=in_channel, num_filters=num_filters, is_stage1=True)
        self.stage2 = ResUNet(in_channel=num_filters + in_channel, num_filters=num_filters)

    def forward(self, x):
        up, out_net_A = self.stage1(x)
        # print(up.shape, out_net_A.shape)
        out_net_B = self.stage2(torch.cat((up, x), dim=1))

        return [out_net_A, out_net_B]


if __name__ == '__main__':
    from torchsummary import summary

    in_channel = 3
    model = CascadeResUNet(in_channel=in_channel).to("cuda")

    summary(model, (in_channel, 128, 128, 128))
