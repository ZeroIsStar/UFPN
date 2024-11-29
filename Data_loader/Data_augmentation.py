import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DataTransform:
    def __init__(self, probability = 0.5,
                 h_flip = True, v_flip = True,  noise = True, scale_random_crop = False,
                 rotate = False, erase = True,
                 mean = 0.0, std = 0.001,
                 sl=0.05, sh=0.35, r1=0.3,
                 ):
        """
           参数：
           "for tensor"
               args: 输入的张量tensor，形状为(c, h, w)。(data,data,.......)
               probability: 执行擦除操作的概率，默认为0.5。
               sl: 擦除区域的最小面积比例，默认为0.05。
               sh: 擦除区域的最大面积比例，默认为0.35。
               r1: 擦除区域的宽高比，默认为0.3。
               mean:随机高斯噪声的均值
               假设你有一个张量图像和其对应的标签
               image = torch.rand(3, 256, 256)
               label = torch.randint(0, 2, (1, 256, 256), dtype=torch.long)  # 256x256大小的标签
               custom_transform = DataTransform(size=(256, 256), v_flip=True, h_flip=True, rotate=True, erase=True, noise=True)
               a, b, label = custom_transform(image, image, label)
               print(a.shape, label.shape)
        """
        self.mean = mean
        self.std  = std
        self.probability = probability
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.rotate = rotate
        self.erase  = erase
        self.scale  = scale_random_crop
        self.noise  = noise
        self.sl     = sl
        self.sh     = sh
        self.r1     = r1

    def __call__(self, *args):
        # Convert tuple to list
        args = list(args)
        list_len = len(args)

        # 随机水平翻转
        if self.h_flip and torch.rand(1) > self.probability:
            for i in range(list_len):
                args[i] = TF.hflip(args[i])

        # 随机垂直翻转
        if self.v_flip and torch.rand(1) > self.probability:
            for i in range(list_len):
                args[i] = TF.vflip(args[i])

        # 随机旋转
        if self.rotate and torch.rand(1) > self.probability:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            for i in range(list_len):
                args[i] = TF.rotate(args[i], angle)

        # 随机擦除
        if self.erase and torch.rand(1) > self.probability:
            c, h, w = args[0].shape
            # 获取擦除区域的参数
            area = h * w
            target_area = torch.randint(int(self.sl * area), int(self.sh * area), size=(1,)).item()
            aspect_ratio = torch.rand(1).item() * self.r1 + 1
            he = int(round((target_area * aspect_ratio) ** 0.5))
            we = int(round((target_area / aspect_ratio) ** 0.5))

            # 随机选择擦除区域的位置
            i = torch.randint(0, h - he, (1,)).item()
            j = torch.randint(0, w - we, (1,)).item()
            erase_area = torch.zeros(c, he, we)

            for index in range(list_len - 1):
                args[index][:, i:i + he, j:j + we] = erase_area
            args[-1][:, i:i + he, j:j + we] = torch.zeros(1, he, we)

        # 放大裁剪
        # rescale
        if self.scale and random.random() > self.probability:
            scale_range = [1, 2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
            c, h, w = args[0].shape
            scale_h, scale_w = int(np.round(h * target_scale)), int(np.round(w * target_scale))
            new_args = args[:]
            for index in range(list_len - 1):
                new_args[index] = F.interpolate(new_args[index].unsqueeze(0), size=(scale_h, scale_w),
                                                mode='bicubic').squeeze(0)
            new_args[-1] = F.interpolate(new_args[-1].unsqueeze(0), size=(scale_h, scale_w), mode='nearest').squeeze(0)
            # crop
            max_h_start = scale_h - h
            max_w_start = scale_w - w
            h_start = torch.randint(0, max_h_start + 1, (1,)).item()
            w_start = torch.randint(0, max_w_start + 1, (1,)).item()
            for index in range(list_len):
                args[index] = new_args[index][:, h_start:h_start + h, w_start:w_start + w]

        # 随机高斯模糊
        if self.noise and random.random() > self.probability:
            std = random.random()*self.std
            for i in range(list_len - 1):
                args[i] = TF.gaussian_blur(args[i], kernel_size=3, sigma=std)
        return args

