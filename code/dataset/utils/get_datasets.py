"""
Helper functions for get datasets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
#import sys
#sys.path.append('../')

def get_datasets(dataset_name, data_root, transforms=None):
    """
    get datasets for train and validation

    Args:
        config (config node object)
    Returns:
        train_set (torch.utils.data.Dataset)
        valid_set (torch.utils.data.Dataset)
    """
    # 如果dataset_name为CFD_train则调用进去
    if dataset_name == 'CFD_train':
        from ..CFD_dataset import CFDDataset#上一目录下的文件
        dataset = CFDDataset(data_root=data_root,split_mode='train',
                               transforms=transforms)
        num_return = dataset.num_return
        
    elif dataset_name == 'CFD_valid':
        from ..CFD_dataset import CFDDataset
        dataset = CFDDataset(data_root=data_root,split_mode='valid',
                               transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'CamCrack810_train':
        from ..CamCrack810_dataset import CamCrack810_Dataset  
        dataset = CamCrack810_Dataset(data_root=data_root, split_mode='train',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'CamCrack810_valid':
        from ..CamCrack810_dataset import CamCrack810_Dataset
        dataset = CamCrack810_Dataset(data_root=data_root, split_mode='valid',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'GAPS384_train':
        from ..GAPS384_dataset import GAPS384_Dataset
        dataset = GAPS384_Dataset(data_root=data_root, split_mode='train',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'GAPS384_valid':
        from ..GAPS384_dataset import GAPS384_Dataset
        dataset = GAPS384_Dataset(data_root=data_root, split_mode='valid',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Fusion_789_GAPS_train':
        from ..Fusion_789_GAPS_dataset import Fusion_789_GAPS_Dataset
        dataset = Fusion_789_GAPS_Dataset(data_root=data_root, split_mode='train',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Fusion_789_GAPS_valid':
        from ..Fusion_789_GAPS_dataset import Fusion_789_GAPS_Dataset
        dataset = Fusion_789_GAPS_Dataset(data_root=data_root, split_mode='valid',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Fusion_789_GAPS_Highway_train':
        from ..Fusion_789_GAPS_Highway_dataset import Fusion_789_GAPS_Highway_Dataset
        dataset = Fusion_789_GAPS_Highway_Dataset(data_root=data_root, split_mode='train',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Fusion_789_GAPS_Highway_valid':
        from ..Fusion_789_GAPS_Highway_dataset import Fusion_789_GAPS_Highway_Dataset
        dataset = Fusion_789_GAPS_Highway_Dataset(data_root=data_root, split_mode='valid',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Fusion_789_Highway_train':
        from ..Fusion_789_Highway_dataset import Fusion_789_Highway_Dataset
        dataset = Fusion_789_Highway_Dataset(data_root=data_root, split_mode='train',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Fusion_789_Highway_valid':
        from ..Fusion_789_Highway_dataset import Fusion_789_Highway_Dataset
        dataset = Fusion_789_Highway_Dataset(data_root=data_root, split_mode='valid',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Fusion_GAPS_Highway_train':
        from ..Fusion_GAPS_Highway_dataset import Fusion_GAPS_Highway_Dataset
        dataset = Fusion_GAPS_Highway_Dataset(data_root=data_root, split_mode='train',
                                transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Fusion_GAPS_Highway_valid':
        from ..Fusion_GAPS_Highway_dataset import Fusion_GAPS_Highway_Dataset
        dataset = Fusion_GAPS_Highway_Dataset(data_root=data_root, split_mode='valid',
                                transforms=transforms)
        num_return = dataset.num_return




    elif dataset_name == 'Highway_train':
        from ..Highway_dataset import Highway_Dataset
        dataset = Highway_Dataset(data_root=data_root, split_mode='train',
                                      transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'Highway_valid':
        from ..Highway_dataset import Highway_Dataset
        dataset = Highway_Dataset(data_root=data_root, split_mode='valid',
                                      transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'CRACK500_Crop_train':
        from ..CRACK500_Crop_dataset import CRACK500CropDataset
        dataset = CRACK500CropDataset(data_root=data_root, split_mode='train',
                                    transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'CRACK500_Crop_test':
        from ..CRACK500_Crop_dataset import CRACK500CropDataset
        dataset = CRACK500CropDataset(data_root=data_root, split_mode='test',
                                    transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'CRACK500_Crop_valid':
        from ..CRACK500_Crop_dataset import CRACK500CropDataset
        dataset = CRACK500CropDataset(data_root=data_root, split_mode='valid',
                                    transforms=transforms)
        num_return = dataset.num_return

    else:
        logging.getLogger('Dataset').error(
            'Dataset for %s not implemented', dataset_name)
        raise NotImplementedError

    return dataset, num_return
