import torch
import torch.nn as nn
from module.FCM import FCM


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


class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        # 第一层
        self.Block1 = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2),
            # FCM(64, 64)
        )
        # 第二层
        self.Block2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2),
            # FCM(128, 128)
        )
        # 第三层
        self.Block3 = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2),
            # FCM(256,256)
        )
        # 第四层
        self.Block4 = nn.Sequential(
            ConvBlock(in_channels=256, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2),
            # FCM(512,512)
        )
        # 第五层
        self.Block5 = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            # ConvBlock(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2),
            # FCM(512, 512)
        )
        # 第六层
        self.Block6 = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=4096),
            nn.Dropout2d()
        )
        # 第七层
        self.Block7 = nn.Sequential(
            ConvBlock(in_channels=4096, out_channels=4096),
            nn.Dropout2d()
        )
        # 第八层
        self.Block8 = nn.Sequential(
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
            # nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x = self.Block5(x)
        x = self.Block6(x)
        x = self.Block7(x)
        x = self.Block8(x)
        return x


# x = torch.randn(8, 3, 512, 512).cuda()
# print(x.shape)
# model = FCN32s(2).cuda()
# y = model(x).cuda()
# print(y.shape)