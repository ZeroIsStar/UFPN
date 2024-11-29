import torch
import torch.nn as nn
from module.Dconv import DconvolutionalBlock
from module.FCM import FCM


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # 第一层
        self.Block1 = nn.Sequential(
            # DconvolutionalBlock(in_channels=3, out_channels=64, kernel_size=3),
            # DconvolutionalBlock(in_channels=64, out_channels=64, kernel_size=3)
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2),  # 64
        )
        # 第二层
        self.Block2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2),  # 64
        )
        # 第三层
        self.Block3 = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2),  # 32 +
        )
        # 第四层
        self.Block4 = nn.Sequential(
            ConvBlock(in_channels=256, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2),  # 16 +
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
            # 回到16 +
        )
        # 第八层
        self.Block8 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            # nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8, padding=0, bias=False)
        )
        self.CT1 = nn.ConvTranspose2d(256, num_classes, kernel_size=1, stride=1)
        self.CT2 = nn.ConvTranspose2d(512, num_classes, kernel_size=1, stride=1)
        self.CT3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x0 = x
        x0 = self.CT1(x0)
        x = self.Block4(x)
        x1 = x
        x1 = self.CT2(x1)
        x = self.Block5(x)
        x = self.Block6(x)
        x = self.Block7(x)
        x2 = x
        x = x1 + x2
        x = self.CT3(x)
        x = x + x0
        x = self.Block8(x)
        return x


if __name__ == "__main__":
    from thop import profile
    import time
    x = torch.randn(1, 3, 512, 512).cuda()
    model = FCN8s(num_classes=2).cuda()
    y = model(x).cuda()
    print(y.shape)
    num_runs = 10
    total_time = 0
    # 多次推理，计算平均推理时间
    for _ in range(num_runs):
        start_time = time.time()
        results = model(x)
        end_time = time.time()
        total_time += (end_time - start_time)
    # 计算平均推理时间
    avg_inference_time = total_time / num_runs
    # 计算FPS
    fps = 1 / avg_inference_time
    print(f"FPS: {fps:.2f} frames per second")
    flops, params = profile(model, inputs=(x,))
    print(f'FLOPs: {flops / 1e9}G')


