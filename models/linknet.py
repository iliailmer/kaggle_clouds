from torch import nn
import math


class ResidualBlock(nn.Module):
    """Two such blocks form one Encoder block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 shortcut=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = shortcut

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.shortcut is not None:
            x += self.shortcut(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4,
                               kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels//4)

        self.relu = nn.ReLU(True)

        self.full_conv = nn.ConvTranspose2d(in_channels//4, in_channels//4,
                                            kernel_size=3, stride=2, padding=1,
                                            bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels//4)

        self.conv3 = nn.Conv2d(in_channels//4, out_channels)
        self.bn3 = nn.BatchNorm2d(in_channels//4)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.full_conv(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        return x


class LinkNet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64,
                               7, 2, 3,
                               bias=False),
        self.bn1 = nn.BatchNorm2d(64),
        self.relu = nn.ReLU(inplace=True)
        self.enc1 = self._make_block(64, 64,
                                     3, stride=1, padding=1)
        self.enc2 = self._make_block(64, 128,
                                     3, stride=2, padding=1)
        self.enc3 = self._make_block(128, 256,
                                     3, stride=2, padding=1)
        self.enc4 = self._make_block(256, 512,
                                     3, stride=2, padding=1)

        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec3 = DecoderBlock(64, 64)

        self.full_conv = nn.ConvTranspose2d(64, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3x3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.out = nn.ConvTranspose2d(32, num_classes, 2, stride=2, bias=False)
        self.bnout = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, in_channels, out_channels,
                    kernel_size, stride, padding, num_res_blocks=1):
        shortcut = None
        layers = []
        strides = [stride] + [1]*num_res_blocks
        for s in strides:
            if in_channels != out_channels or s != 1:
                shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1,
                              stride, padding, bias=False),
                    nn.BatchNorm2d(out_channels))
            layers.append(ResidualBlock(in_channels, out_channels,
                                        kernel_size, s, padding, shortcut))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec4 = self.dec4(enc4) + enc3
        dec3 = self.dec3(dec4) + enc2
        dec2 = self.dec2(dec3) + enc1
        dec1 = self.dec1(dec2)

        full_conv = self.relu(self.bn2(self.full_conv(dec1)))
        conv3x3 = self.relu(self.bn3(self.conv3x3(full_conv)))
        out = self.relu(self.bnout(self.out(conv3x3)))

        return out

    def freeze(self, modules):
        assert isinstance(modules, list), "Modules must be passed as a list."
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(False)
