from pytorch_toolbelt.modules import encoders
import math
from torch import nn
import torch
from .swish import Swish


class Attention(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, out_channels):
        super().__init__()

        self.conv_g = Conv2dBn(in_channels_g, out_channels,
                               1, 1, 0, True)  # g= shortcut connection
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        self.conv_x = Conv2dBn(in_channels_x, out_channels,
                               1, 1, 0, True)
        self.activ = nn.ReLU(True)  # Swish()
        self.psi = nn.Sequential(Conv2dBn(out_channels, 1, 1,1, 0, True),
            nn.Sigmoid())

    def forward(self, x_, g):
        x = self.up(x_)
        x = self.conv_x(x)
        g = self.psi(self.activ(self.conv_g(g)+x))
        x = x*g
        return x


class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class DecoderBlockGated(nn.Module):
    def __init__(self, in_channels, in_channels_shortcut, out_channels):
        super().__init__()
        self.convbn1 = Conv2dBn(in_channels, in_channels // 4,
                                k=3, s=1, p=1)
        self.act = nn.ReLU(True)
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        self.attention_upsample = Attention(in_channels // 4,
                                            in_channels_shortcut,
                                            out_channels=out_channels)
                                            # in_channels // 4)

        self.convbn2 = Conv2dBn(in_channels//4, out_channels,
                                k=1, s=1, p=0)

    def forward(self, x, shortcut):
        in_x = self.up(x)
        x = self.act(self.convbn1(x))
        x = self.attention_upsample(x, shortcut)
        # x = self.act(self.convbn2(x))
        return torch.cat([x, in_x], dim=1)


class LinkNetGated(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        model = encoders.Resnet34Encoder()

        self.conv1_ = model.layer0[0]
        # if in_channels == 3 else nn.Conv2d(in_channels,
        #                                    64, 7, 2, 3)
        self.bn1 = model.layer0[1]
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = model.layer1  # 64
        self.enc2 = model.layer2  # 128
        self.enc3 = model.layer3  # 256
        self.enc4 = model.layer4  # 512

        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

        self.dec3 = DecoderBlockGated(512, 256, 256)
        self.conv3 = Conv2dBn(512+256, 256, 3, 1, 1)  # 256
        self.dec2 = DecoderBlockGated(256, 128, 128)
        self.conv2 = Conv2dBn(256+128, 128, 3, 1, 1)  # 128
        self.dec1 = DecoderBlockGated(128, 64, 64)
        self.conv1 = Conv2dBn(128+64, 64, 3, 1, 1)  # 64

        self.full_conv = DecoderBlockGated(64, 64, 32)

        self.convbn3 = Conv2dBn(32+64, 16, 3, 1, 1)
        self.out = nn.Sequential(nn.Upsample(scale_factor=2,
                                             mode='bilinear',
                                             align_corners=True),
                                             nn.Conv2d(16, num_classes, 2,
                                                       stride=2, bias=False))

        for (name, m) in self.named_modules():
            if 'enc' not in name and 'conv1_' not in name:
                if isinstance(m, nn.Conv2d) or isinstance(m,
                                                          nn.ConvTranspose2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        x_ = self.act(self.bn1(self.conv1_(x)))

        x = self.pool(x_)

        enc1 = self.enc1(x)    # 64
        enc2 = self.enc2(enc1)  # 128
        enc3 = self.enc3(enc2)  # 256
        enc4 = self.enc4(enc3)  # 512

        dec3 = self.act(self.conv3(self.dec3(enc4, enc3)))
        dec2 = self.act(self.conv2(self.dec2(dec3, enc2)))
        dec1 = self.act(self.conv1(self.dec1(dec2, enc1)))
        conv3x3 = self.act(self.convbn3(self.full_conv(dec1, x_)))
        out = self.act(self.out(conv3x3))

        return out