import torch
import torch.nn as nn
import torch.nn.functional as F


class Double_conv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,kernel_size=(3, 3),padding=1,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Down, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.conv = Double_conv(in_planes, out_planes)
        self.conv1X1 = nn.Conv2d(in_planes*2, in_planes, kernel_size=1)

    def forward(self, x):
        max = self.max_pool(x)
        avg = self.avg_pool(x)
        mix = torch.cat([max, avg], dim=1)
        x = self.conv1X1(mix)
        return self.conv(x)


class Feature_Pyramid(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Feature_Pyramid, self).__init__()
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

    def forward(self, high_layer, low_layer):
        low_layer = F.interpolate(low_layer, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv1x1(high_layer) + low_layer
        return out


class model(nn.Module):
    def __init__(self, in_planes, num_classes=21):
        super(model, self).__init__()
        self.dim = [64, 128, 256,512, 1024]
        self.conv = Double_conv(in_planes, self.dim[0])
        self.down1 = Down(self.dim[0], self.dim[1])
        self.down2 = Down(self.dim[1], self.dim[2])
        self.down3 = Down(self.dim[2], self.dim[3])
        self.down4 = Down(self.dim[3], self.dim[4])

        self.conv1x1 = nn.Conv2d(self.dim[4], self.dim[1], kernel_size=1)
        self.fpn4 = Feature_Pyramid(self.dim[3], self.dim[1])
        self.fpn3 = Feature_Pyramid(self.dim[2], self.dim[1])
        self.fpn2 = Feature_Pyramid(self.dim[1], self.dim[1])
        self.fc = nn.Conv2d(self.dim[1]*4, self.dim[0], kernel_size=1)

        self.conv3x3_GN_GELU = nn.Sequential(
            nn.Conv2d(self.dim[1], self.dim[1], kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(16, self.dim[1]),
            nn.GELU()
        )

        self.Classifier = nn.Sequential(
            nn.Conv2d(self.dim[0] * 2, self.dim[0], kernel_size=1),
            nn.GroupNorm(16, self.dim[0]),
            nn.GELU(),
            nn.Conv2d(self.dim[0], self.dim[0], kernel_size=3, padding=1),
            nn.Conv2d(self.dim[0], num_classes, kernel_size=1),
        )

    def forward(self, x):
        layer1 = self.conv(x)
        layer2 = self.down1(layer1)
        layer3 = self.down2(layer2)
        layer4 = self.down3(layer3)
        layer5 = self.down4(layer4)

        p5 = self.conv3x3_GN_GELU(self.conv1x1(layer5))
        p4 = self.conv3x3_GN_GELU(self.fpn4(layer4, p5))
        p3 = self.conv3x3_GN_GELU(self.fpn3(layer3, p4))
        p2 = self.conv3x3_GN_GELU(self.fpn2(layer2, p3))

        p5 = F.interpolate(p5, scale_factor=8, mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.fc(torch.cat([p5, p4, p3, p2], dim =1))
        return self.Classifier(torch.cat([F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True), layer1], dim=1))


if __name__ == "__main__":
    from thop import profile
    import time
    model = model(3, num_classes=2).cuda()
    x = torch.rand(1, 3, 512, 512).cuda()
    output = model(x)
    flops, params = profile(model, inputs=(x,))
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
    print(f'FLOPs: {flops / 1e9}G')
    print(f'params: {params / 1e6}M')
