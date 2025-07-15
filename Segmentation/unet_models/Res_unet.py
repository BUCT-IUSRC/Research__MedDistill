import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

class ResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        filters = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = ResidualBlock(in_ch, filters[0])
        self.conv2 = ResidualBlock(filters[0], filters[1])
        self.conv3 = ResidualBlock(filters[1], filters[2])
        self.conv4 = ResidualBlock(filters[2], filters[3])
        self.conv5 = ResidualBlock(filters[3], filters[4])

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.up_conv4 = ResidualBlock(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.up_conv3 = ResidualBlock(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up_conv2 = ResidualBlock(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.up_conv1 = ResidualBlock(filters[1], filters[0])

        self.final_conv = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        d4 = self.up4(x5)
        d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.up_conv4(torch.cat([x4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up_conv3(torch.cat([x3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up_conv2(torch.cat([x2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.up_conv1(torch.cat([x1, d1], dim=1))

        out = self.final_conv(d1)
        return out

def get_resunet():
    return ResUNet()
