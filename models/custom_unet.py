from torch import nn
import torch
from torchvision import models


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=k, stride=s, padding=p,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self._weight_init()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

    def _weight_init(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CenterBlock(nn.Module):
    def __init__(self, in_channels, mid, out_channels):
        super().__init__()
        self.conv3x3 = ConvReLU(in_channels, mid,
                                k=3, s=2, p=1)
        self.conv1x1 = ConvReLU(mid, out_channels,
                                k=1, s=1, p=0)
        self._weight_init()

    def _weight_init(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 deconv=True, shape=None, kwargs=None):
        super().__init__()
        self.conv3x3 = ConvReLU(in_channels, mid_channels)
        self.conv1x1 = ConvReLU(mid_channels, out_channels,
                                k=1, s=1, p=0)
        if deconv:
            self.up = nn.Sequential(nn.ConvTranspose2d(mid_channels,
                                                       mid_channels,
                                                       kernel_size=4,
                                                       stride=2,
                                                       padding=1,
                                                       bias=False),
                                    nn.BatchNorm2d(mid_channels)
                                    )

        else:
            self.up = nn.Upsample(**kwargs)

        self.relu = nn.ReLU()
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv3x3(x)

        x = self.up(x)

        x = self.conv1x1(x)
        return x


class UNetResNet(nn.Module):
    def __init__(self, pretrained=True, deconv=True, upsample_args=None):
        super().__init__()
        _input = ConvReLU(3, 64, 7, 2, 3)

        net = models.resnet18(pretrained)
        self.encoder = nn.Sequential(_input,
                                     net.layer1,
                                     net.layer2,
                                     net.layer3,
                                     net.layer4)
        self._center = CenterBlock(512, 512, 512)
        self.decoder = nn.Sequential(UpsampleBlock(512, 256, 512,
                                                   deconv=deconv,
                                                   kwargs=upsample_args),
                                     UpsampleBlock(1024, 256, 256,
                                                   deconv=deconv,
                                                   kwargs=upsample_args),
                                     UpsampleBlock(512, 64, 128,
                                                   deconv=deconv,
                                                   kwargs=upsample_args),
                                     UpsampleBlock(256, 32, 64,
                                                   kwargs=upsample_args,
                                                   deconv=deconv),
                                     ConvReLU(128, 64))
        self.finish = ConvReLU(64, 32)
        self.out = ConvReLU(32, 4, 1, p=0)

        # self._weight_init(_input)
        # self._weight_init(self._center)
        if pretrained is False:
            self._weight_init(self.encoder)
        # self._weight_init(self.decoder)

    def _weight_init(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze(self, modules):
        assert isinstance(modules, list), "Modules must be passed as a list."
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        # input is 3x320x640
        x = self.encoder[0](x)  # 64x320x640
        l1 = self.encoder[1](x)  # 64x320x640
        l2 = self.encoder[2](l1)  # 128x160x320
        l3 = self.encoder[3](l2)  # 256x80x160
        l4 = self.encoder[4](l3)  # 512x40x80

        c = self._center(l4)  # 512x20x40

        up4 = self.decoder[0](c)  # 512x40x80
        up3 = self.decoder[1](torch.cat([up4, l4], dim=1))  # 256x80x160
        up2 = self.decoder[2](torch.cat([up3, l3], dim=1))  # 128x160x320
        up1 = self.decoder[3](torch.cat([up2, l2], dim=1))  # 64x320x640

        finish = self.decoder[4](torch.cat([up1, l1], dim=1))  # 64x320x640
        out = self.out(self.finish(finish))  # 32x320x640 -> 4x320x640
        return out
