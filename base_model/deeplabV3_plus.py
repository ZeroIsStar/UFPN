import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50
from model.Attention.ShuffleAttention import ShuffleAttention
from module.FCM import FCM


class DeepLabV3plus(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, dilation = [6,12,18]):
        super(DeepLabV3plus, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone(in_channels=in_channels)
        self.aspp = ASPP(256, 512, dilation=dilation)
        self.decoder = Decoder(512, num_classes=num_classes)
        self.conv1x1 = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, x):
        x, low_level_feature = self.backbone(x)
        aspp = self.aspp(x)
        x = self.decoder(low_level_feature, aspp)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation =[6,12,18]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation[0], padding=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation= dilation[1], padding=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation= dilation[2], padding=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(32),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.global_pool(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Decoder, self).__init__()
        self.aspp_conv1x1 = nn.Conv2d(in_channels*5, out_channels=in_channels, kernel_size=1)
        self.conv1x1 = nn.Conv2d(in_channels//8, out_channels=in_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        self.conv3x3 = nn.Conv2d(2 * in_channels, num_classes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low_features, aspp_features):
        aspp1 = self.aspp_conv1x1(aspp_features)
        aspp1 = self.relu(self.up(aspp1))
        low_features = self.conv1x1(low_features)

        concat   = torch.cat((low_features, aspp1), dim=1)
        concat_cov = self.conv3x3(concat)
        return self.up(concat_cov)


class backbone(nn.Module):
    def __init__(self, in_channels, backbone_model = None, pretrained=None):
        super(backbone, self).__init__()
        if backbone_model is None:
            self.backbone_model = resnet34(weights=pretrained)
        elif backbone_model =='resnet50':
            self.backbone_model = resnet50(weights=pretrained)

        else:
            self.backbone_model = backbone_model
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.backbone_model.layer1
        self.layer2 = self.backbone_model.layer2
        self.layer3 = self.backbone_model.layer3
        self.layer4 = self.backbone_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        low_level_feature = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x, low_level_feature


if __name__ == "__main__":
    from thop import profile
    import time
    x = torch.randn(1, 3, 512, 512).cuda()
    model = DeepLabV3plus(in_channels=3, num_classes=2).cuda()
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
    print(f'Params: {params / 1e6}M')
    print(model(x).shape)

