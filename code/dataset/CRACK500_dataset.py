"""
CRACK500 dataset class implemented for pytorch
Download the dataset from: https://drive.google.com/open?id=1y9SxmmFVh0xdQR-wdchUmnScuWMJ5_O-
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging
from zipfile import ZipFile
import time
from os import path
from PIL import Image
import PIL.ImageOps
from torch.utils.data import Dataset
from torch.utils.data import Subset

from torchvision.transforms import functional as TF
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import list_files

class CRACK500Dataset(Dataset):
    """
    CRACK500 dataset with augmentation and formatting
    """

    def __init__(self, data_root, split_mode,
                 transforms=None, download=False, extract=False):
        self.data_root = data_root
        self.split_mode = split_mode
        self.transforms = transforms

        self.logger = logging.getLogger('CRACK500 Dataset')
        self.logger.debug('split_mode is set to %s', self.split_mode)
        
        self.dataset = CRACK500PILDataset(data_root=self.data_root, split_mode=self.split_mode)
           
        if download or extract == True:
            self.logger.error('please download the dataset on https://drive.google.com/open?id=1y9SxmmFVh0xdQR-wdchUmnScuWMJ5_O-')
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


    @staticmethod
    def _default_trans(image, annot, split_mode):

        image = TF.resize(image, size=(512, 512))
        annot = TF.resize(annot, size=(512, 512))

        if split_mode == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
                annot = TF.hflip(annot)

            if random.random() > 0.5:
                image = TF.vflip(image)
                annot = TF.vflip(annot)

        elif split_mode == 'valid' or split_mode == 'test':
            pass

        image = TF.to_tensor(image)
        annot = TF.to_tensor(annot)

        return image, annot


class CRACK500PILDataset(Dataset):
    """
    CRACK500 dataset of original PIL images
    """

    def __init__(self, data_root, split_mode):
        self.logger = logging.getLogger('CRACK500 PIL Dataset')
        self.data_root = path.expanduser(data_root)
        self.split_mode = split_mode

        if self.split_mode == 'valid':
            self._dir = path.join(self.data_root, 'CRACK500', 'valdata')
        elif self.split_mode == 'train':
            self._dir = path.join(self.data_root, 'CRACK500', 'traindata')
        elif self.split_mode == 'test':
            self._dir = path.join(self.data_root, 'CRACK500', 'testdata')
        
        self._image_paths = sorted(list_files(self._dir,suffix=('.jpg', '.JPG'),prefix=True))
        self._annot_paths = sorted(list_files(self._dir,suffix=('.png', '.PNG'),prefix=True))
        
        assert len(self._image_paths) == len(
            self._annot_paths), 'CRACK500 dataset corrupted'
        
        self.start = time.time()
        self.logger.debug('Found all %d samples for CRACK500 dataset',
                          len(self._image_paths))

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        annot = Image.open(self._annot_paths[index], mode='r').convert('1')
        #print(' get dataset %d/%d || cost %.3f s \r'%(index+1,len(self._image_paths),time.time()-self.start),end='')
        assert image.size==annot.size, self._image_paths[index]+'\n'+self._annot_paths[index]
        return image, annot

    def __len__(self):
        return len(self._image_paths)

if __name__ == '__main__':
    leaveout_ids = [idx for idx in range(8)]
    CRACK500_dataset = CRACK500Dataset('../data/', 'test',transforms=None, download=False, extract=False)
    #CRACK500_dataset = CRACK500PILDataset('../data/','train')
    import matplotlib.pyplot as plt
    import torch
#    plt.ion()
    for i in range(len(CRACK500_dataset)):
        a=CRACK500_dataset[i][1]
#        plt.imshow(torch.squeeze(CRACK500_dataset[i][1]).numpy())
#        plt.pause(0.1)
#    plt.show()
#    from PIL import Image
#    a = Image.open('../data/CRACK500/traindata/20160222_081011.jpg').convert('RGB')
#    b = Image.open('../data/CRACK500/traindata/20160222_081011_mask.png').convert('1')
#    print(a.size)
#    print(b.size)