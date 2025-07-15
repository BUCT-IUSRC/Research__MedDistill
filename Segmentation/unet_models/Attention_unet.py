import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 尺寸对齐
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        filters = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = ConvBlock(in_ch, filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.conv4 = ConvBlock(filters[2], filters[3])
        self.conv5 = ConvBlock(filters[3], filters[4])

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.up_conv4 = ConvBlock(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.up_conv3 = ConvBlock(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.up_conv2 = ConvBlock(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.up_conv1 = ConvBlock(filters[1], filters[0])

        self.final_conv = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        d4 = self.up4(x5)
        d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x4 = self.att4(g=d4, x=x4)
        d4 = self.up_conv4(torch.cat([x4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x3 = self.att3(g=d3, x=x3)
        d3 = self.up_conv3(torch.cat([x3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x2 = self.att2(g=d2, x=x2)
        d2 = self.up_conv2(torch.cat([x2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.att1(g=d1, x=x1)
        d1 = self.up_conv1(torch.cat([x1, d1], dim=1))

        out = self.final_conv(d1)
        return out

def get_attention_unet():
    return AttentionUNet()
