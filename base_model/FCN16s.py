import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FCN16s(nn.Module):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()

        # 第一层
        self.Block1 = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第二层
        self.Block2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第三层
        self.Block3 = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第四层
        self.Block4 = nn.Sequential(
              # 16 +
            ConvBlock(in_channels=256, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第五层
        self.Block5 = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2),  # 8
        )
        # 第六层
        self.Block6 = nn.Sequential(
            # 8
            ConvBlock(in_channels=512, out_channels=4096),
            nn.Dropout2d()
        )
        # 第七层
        self.Block7 = nn.Sequential(
            nn.ConvTranspose2d(4096, num_classes, kernel_size=2, stride=2),
        )
        # 第八层
        self.Block8 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
            # nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=16, padding=0, bias=False)
        )
        self.CT1 = nn.ConvTranspose2d(512, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x0 = x
        x0 = self.CT1(x0)
        x = self.Block5(x)
        x = self.Block6(x)
        x = self.Block7(x)
        x1 = x
        x = x0 + x1
        x = self.Block8(x)
        return x


# x = torch.randn(8, 3, 256, 256).cuda()
# print(x.shape)
# model = FCN16s(2).cuda()
# y = model(x).cuda()
# print(y.shape)