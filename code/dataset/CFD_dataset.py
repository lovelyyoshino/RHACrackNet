"""
CFD dataset class implemented for pytorch
Download the dataset from: https://drive.google.com/open?id=1y9SxmmFVh0xdQR-wdchUmnScuWMJ5_O-
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging


from os import path
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.transforms import functional as TF
from torchvision.datasets.utils import list_files

class CFDDataset(Dataset):
    """
    CFD dataset with augmentation and formatting
    """

    def __init__(self, data_root, split_mode,transforms=None):
        self.data_root = data_root#data文件
        self.split_mode = split_mode#是训练还是评估
        self.transforms = transforms
        self.num_return = 2

        self.logger = logging.getLogger('CFD Dataset')
        self.logger.debug('split_mode is set to %s', self.split_mode)#输出log


        if self.split_mode == 'train':
            self.dataset, _ = self._get_subsets()
        elif self.split_mode == 'valid':
            _, self.dataset = self._get_subsets()
        else:
            self.logger.error('split_mode must be either "train" or "valid"')
            raise NotImplementedError
            

    def __getitem__(self, index):
        image, annot = self.dataset[index]

        if self.transforms is None:
            image, annot = self._default_trans(
                image, annot, split_mode=self.split_mode)

        else:
            image, annot = self.transforms(
                image, annot, split_mode=self.split_mode)

        return image, annot

    def __len__(self):
        return len(self.dataset)

    def _get_subsets(self):
        pil_dataset = CFDPILDataset(data_root=self.data_root)#初始化CFDPILDataset
        num_samples = len(pil_dataset)#获得数目

        try:
            train_ids = [idx for idx in range(72)]#训练样本72个

            valid_ids = [idx for idx in range(72,num_samples)]#剩下的为测试样本

            train_set = Subset(pil_dataset, train_ids)#分割整体数据集
            valid_set = Subset(pil_dataset, valid_ids)

        except:
            self.logger.error('invalid leaveout_ids: %s')
            raise ValueError('invalid leaveout_ids: %s')

        return train_set, valid_set

    @staticmethod
    def _default_trans(image, annot, split_mode):

#        image = TF.resize(image, size=(512, 512))
#        annot = TF.resize(annot, size=(512, 512))

        if split_mode == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)#对图像进行翻转
                annot = TF.hflip(annot)

            if random.random() > 0.5:
                image = TF.vflip(image)
                annot = TF.vflip(annot)

        elif split_mode == 'valid':
            pass

        image = TF.to_tensor(image)
        annot = TF.to_tensor(annot)

        return image, annot


class CFDPILDataset(Dataset):#原始PIL图像的CFD数据集
    """
    CFD dataset of original PIL images
    """

    def __init__(self, data_root):
        self.logger = logging.getLogger('CFD PIL Dataset')
        self.data_root = path.expanduser(data_root)


        self._image_dir = path.join(self.data_root, 'CFD', 'cfd_image')#导入对应的路径
        self._annot_dir = path.join(self.data_root, 'CFD', 'seg_gt')
        self._image_paths = sorted(list_files(self._image_dir,suffix=('.jpg', '.JPG'),prefix=True))#导入原始文件并进行排序（001.jpg）
        self._annot_paths = sorted(list_files(self._annot_dir,suffix=('.png', '.PNG'),prefix=True))#导入分割文件并进行排序（001.png）

        assert len(self._image_paths) == len(
            self._annot_paths), 'CFD dataset corrupted'#断言触发，防止数目不对等

        self.logger.debug('Found all %d samples for CFD dataset',
                          len(self._image_paths))

    def __getitem__(self, index):#打开图片
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        annot = Image.open(self._annot_paths[index], mode='r').convert('1')
        return image, annot

    def __len__(self):
        return len(self._image_paths)

if __name__ == '__main__':
    leaveout_ids = [idx for idx in range(8)]
    CFD_dataset = CFDDataset('../Datasets/', 'valid', leaveout_ids,transforms=None, download=False, extract=False)
#    import matplotlib.pyplot as plt
#    import torch
#    plt.ion()
#    for i in range(len(CFD_dataset)):
#        plt.imshow(torch.squeeze(CFD_dataset[i][1]).numpy())
#        plt.pause(0.1)
#    plt.show()
