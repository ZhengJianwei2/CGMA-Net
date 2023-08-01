from lib.bkbone.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
from CGMA.model_util import SpatialAttention, ChannelAttention, PSPModule


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=7, stride=1, padding=3):
        super(DWConv, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=input_dim)
        self.conv2 = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1), stride=1, padding=0)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LE(nn.Module):
    def __init__(self, input_dim, out_dim, scale):
        super(LE, self).__init__()
        self.dw1 = DWConv(input_dim, out_dim)
        self.dw2 = DWConv(out_dim, out_dim)

    def forward(self, x):
        x = self.dw1(x)
        x = self.dw2(x)
        return x


class MAD(nn.Module):
    def __init__(self, channel, out_channel, d2=7, d3=5):
        super(MAD, self).__init__()

        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.dw_conv1 = BasicConv2d(channel, channel, 3, padding=1, dilation=1)
        self.dw_conv2 = BasicConv2d(channel, channel, 3, padding=d3, dilation=d3)
        self.dw_conv3 = BasicConv2d(channel, channel, 3, padding=d2, dilation=d2)
        self.conv_fuse1 = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.conv_fuse2 = BasicConv2d(channel * 2, out_channel, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        b1 = self.dw_conv1(x1)
        b2 = self.dw_conv2(x1)
        b3 = self.dw_conv3(x1)
        f1 = self.conv_fuse1(torch.cat([b1, b2], 1))
        f2 = self.conv_fuse2(torch.cat([f1, b3], 1))
        return f2


class Edge_Extract(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Edge_Extract, self).__init__()

        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)
        self.conv_out = nn.Conv2d(out_channel, 1, 3, 1, 1)
        self.ca = ChannelAttention(out_channel)
        self.sa = SpatialAttention()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        x = self.ca(x) * x  # channel attention
        x = self.sa(x) * x  # spatial attention


        return x


class CGMA(nn.Module):
    def __init__(self, channel=32):
        super(CGMA, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'bkbone/pvt_v2_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1_1 = LE(64, channel, 1)
        self.Translayer2_1 = LE(128, channel, 2)
        self.Translayer3_1 = LE(320, channel, 4)
        self.Translayer4_1 = LE(512, channel, 8)
        self.Translayer4_2 = torch.nn.Conv2d(512, channel, 1)

        self.decoder4 = nn.Sequential(
            MAD(channel * 2, channel, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        self.sa4 = SpatialAttention()

        self.decoder3 = nn.Sequential(
            MAD(channel * 3, channel, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        self.sa3 = SpatialAttention()

        self.decoder2 = nn.Sequential(
            MAD(channel * 4, channel, 5, 3),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        self.sa2 = SpatialAttention()

        self.decoder1 = nn.Sequential(
            MAD(channel * 5, channel, 5, 3),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S1 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        self.sa1 = SpatialAttention()

        self.decoder0 = nn.Sequential(
            MAD(channel * 2, channel, 5, 3),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S0 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.edge_extract = Edge_Extract(64, channel)
        self.ppm = PSPModule(512)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        # CIM
        x1_t = self.Translayer1_1(x1)
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        sa_map4 = self.upsample(self.sa4(x4_t))
        sa_map3 = self.upsample(self.sa3(x3_t))
        sa_map2 = self.upsample(self.sa2(x2_t))

        aspp = self.Translayer4_2(self.ppm(x4))
        f4 = self.decoder4(torch.cat([x4_t, aspp], dim=1))
        # sa_map4 = self.sa4(f4)
        out4 = self.S4(f4)

        f3 = torch.cat([x3_t, x3_t * sa_map4, f4], dim=1)
        f3 = self.decoder3(f3)
        # sa_map3 = self.sa3(f3)
        out3 = self.S3(f3)

        f2 = torch.cat([x2_t, x2_t * self.upsample(sa_map4), x2_t * sa_map3, f3], dim=1)
        f2 = self.decoder2(f2)
        # sa_map2 = self.sa2(f2)
        out2 = self.S2(f2)

        f1 = torch.cat([x1_t, x1_t * sa_map2, x1_t * self.upsample(sa_map3),
                        x1_t * F.interpolate(sa_map4, scale_factor=4, mode='bilinear'), f2], dim=1)
        f1 = self.decoder1(f1)
        sa_map1 = self.sa1(f1)
        out1 = self.S1(f1)

        fea = self.edge_extract(x1)
        fea = F.interpolate(fea, scale_factor=2, mode='bilinear')

        detail = fea * sa_map1
        body = f1 + fea

        out0 = self.S0(self.decoder0(torch.cat([detail, body], dim=1)))
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear')
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear')
        out3 = F.interpolate(out3, scale_factor=8, mode='bilinear')
        out4 = F.interpolate(out4, scale_factor=16, mode='bilinear')

        out = []
        out.append(out0)
        out.append(out1)
        out.append(out2)
        out.append(out3)
        out.append(out4)

        return out

