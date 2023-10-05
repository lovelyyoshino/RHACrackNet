"""
CFD dataset class implemented for pytorch
Download the dataset from: https://drive.google.com/open?id=1y9SxmmFVh0xdQR-wdchUmnScuWMJ5_O-
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging
import numpy as np


from os import path
from PIL import Image
from PIL import ImageEnhance
import torchvision.transforms as TT
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.transforms import functional as TF
from torchvision.datasets.utils import list_files


class CamCrack810_Dataset(Dataset):
    """
    CamCrack810 dataset with augmentation and formatting
    """

    def __init__(self, data_root, split_mode,transforms=None):
        self.data_root = data_root#data文件
        self.split_mode = split_mode#是训练还是评估
        self.transforms = transforms
        self.num_return = 2

        self.logger = logging.getLogger('CamCrack810 Dataset')
        self.logger.debug('split_mode is set to %s', self.split_mode)#输出log


        if self.split_mode == 'train':
            train_data_root = path.join(self.data_root, 'CamCrack810', 'train')
            self.dataset = CamCrack810PILDataset(data_root=train_data_root)
        elif self.split_mode == 'valid':
            valid_data_root = path.join(self.data_root, 'CamCrack810', 'valid')
            self.dataset = CamCrack810PILDataset(data_root=valid_data_root)
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







    '''
        考虑到GT的变换要和增强的图片一致，这里没有使用
        transforms.RandomApply，或 transforms.RandomRotation
    '''
    @staticmethod
    def _default_trans(image, annot, split_mode):

#        image = TF.resize(image, size=(256, 256))
#        annot = TF.resize(annot, size=(256, 256))

        if split_mode == 'train':

            # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # annot = cv2.cvtColor(np.array(annot), cv2.COLOR_RGB2BGR)
            # # 对图像进行垂直翻转
            # if random.random() > 0.5:
            #     image = h_flip(image)
            #     annot = h_flip(annot)
            # # 对图像进行水平翻转
            # if random.random() > 0.5:
            #     image = v_flip(image)
            #     annot = v_flip(annot)
            # # 对图像进行角度旋转
            # if random.random() > 0.5:
            #     image = dst(image)
            #     annot = dst(annot)
            # # 调整图像的亮度
            # if random.random() > 0.5:
            #     image = light_up(image)
            # # 调整图像的亮度
            # if random.random() > 0.5:
            #     image = light_down(image)
            # # 调整图像的亮度
            # if random.random() > 0.5:
            #     image = contrast_change(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # annot = cv2.cvtColor(annot, cv2.COLOR_BGR2RGB)

            # 对图像进行垂直翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                annot = TF.hflip(annot)
            # 对图像进行水平翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                annot = TF.vflip(annot)
            # 对图像进行翻转180
            if random.random() > 0.5:
                image = TF.hflip(image)
                image = TF.vflip(image)
                annot = TF.hflip(annot)
                annot = TF.vflip(annot)
            # 对图像亮度增强
            if random.random() > 0.5:
                enh_bri = ImageEnhance.Brightness(image)
                brightness = 1.5
                image = enh_bri.enhance(brightness)
            # 对图像亮度减弱
            if random.random() > 0.5:
                enh_bri = ImageEnhance.Brightness(image)
                brightness = 0.5
                image = enh_bri.enhance(brightness)
            # 对图像对比度增强
            if random.random() > 0.5:
                enh_con = ImageEnhance.Contrast(image)
                contrast = 1.5
                image = enh_con.enhance(contrast)

            # # 对图像进行随机角度旋转
            # if random.random() > 0.5:
            #     angle = random.randint(0, 360)
            #     image = TF.rotate(image, angle)
            #     annot = TF.rotate(annot, angle)
            # # 对比度变化
            # if random.random() > 0.5:
            #     num_contrast = random.random()
            #     image = TT.ColorJitter(contrast=num_contrast)
            #     # GT不用变换
            # # 调整图像的亮度
            # if random.random() > 0.5:
            #     num_bright = random.random()
            #     image = TT.ColorJitter(brightness=num_bright)
            # # 调整图像的色相
            # if random.random() > 0.5:
            #     num_hue= 0.3 # random.random() % 0.5
            #     image = TT.ColorJitter(hue=num_hue)

        elif split_mode == 'valid':
            pass

        image = TF.to_tensor(image)
        annot = TF.to_tensor(annot)

        return image, annot


class CamCrack810PILDataset(Dataset):#原始PIL图像的CFD数据集
    """
    CFD dataset of original PIL images
    """

    def __init__(self, data_root):
        self.logger = logging.getLogger('CamCrack810 PIL Dataset')
        self.data_root = path.expanduser(data_root)


        self._image_dir = path.join(self.data_root, 'image')#导入对应的路径
        self._annot_dir = path.join(self.data_root, 'seg_gt')
        self._image_paths = sorted(list_files(self._image_dir,suffix=('.png', '.PNG'),prefix=True))#导入原始文件并进行排序（001.jpg）
        self._annot_paths = sorted(list_files(self._annot_dir,suffix=('.png', '.PNG'),prefix=True))#导入分割文件并进行排序（001.png）

        assert len(self._image_paths) == len(
            self._annot_paths), 'CamCrack810 dataset corrupted'#断言触发，防止数目不对等

        self.logger.debug('Found all %d samples for CamCrack810 dataset',
                          len(self._image_paths))

    def __getitem__(self, index):#打开图片
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        annot = Image.open(self._annot_paths[index], mode='r').convert('1')
        return image, annot

    def __len__(self):
        return len(self._image_paths)

if __name__ == '__main__':
    leaveout_ids = [idx for idx in range(8)]
    CamCrack810_dataset = CamCrack810_Dataset('../Datasets/', 'valid', leaveout_ids,transforms=None, download=False, extract=False)
#    import matplotlib.pyplot as plt
#    import torch
#    plt.ion()
#    for i in range(len(CFD_dataset)):
#        plt.imshow(torch.squeeze(CFD_dataset[i][1]).numpy())
#        plt.pause(0.1)
#    plt.show()
