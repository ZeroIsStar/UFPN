import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from Data_loader.Data_augmentation import DataTransform


class landslide_Dataset(data.dataloader.Dataset):
    def __init__(self, dir = None, set = None, h_flip=True, v_flip=True, scale_random_crop=False, noise=False, rotate = False, erase = True):
        """
        :param dir: dataset directory
        :param set:['train', 'val', 'test']
        """
        self.files = []
        self.set = set
        self.transform = DataTransform(h_flip=h_flip, v_flip=v_flip, scale_random_crop=scale_random_crop, noise=noise, rotate = rotate, erase = erase)
        self.img_dir = os.path.join(dir, "img")
        self.mask_dir = os.path.join(dir, "mask")
        with open(dir + '\\' + set + '.txt', 'r') as file:
            # 读取文件中的所有内容
            data_name = [line.strip() for line in file]
        self.files = []
        for name in data_name:
            img_dir = os.path.join(self.img_dir, name)
            mask_dir = os.path.join(self.mask_dir, name)
            self.files.append({
                "img": img_dir,
                "mask": mask_dir,
                "name": name
            })

    def __len__(self):
        # 返回数据集的长度
        return len(self.files)

    def normalize(self, data):
        min = data.min()
        max = data.max()
        x = (data - min) / (max - min)
        return x

    def __getitem__(self, index):
        datas = self.files[index]
        name = datas['name']
        img = Image.open(datas['img'])
        label = Image.open(datas['mask'])
        img = torch.as_tensor(np.array(img, np.float32).transpose((-1, 0, 1)))
        label = torch.as_tensor(np.array(label), dtype=torch.uint8)
        # 数据加强
        if self.set == 'train':
            label = label.unsqueeze(0)
            img, label = self.transform(img, label)
            label = label.squeeze(0)
        img = self.normalize(img)
        label = label.long()
        return img, label, name


# a = landslide_Dataset(dir= 'E:\\CAS_Landslide\\Wenchuan', set='train')
# img, label, name = a.__getitem__(0)
# print(img.shape,label.shape)
