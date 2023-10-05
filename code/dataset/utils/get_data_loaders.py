"""
Helper functions for get data loaders
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader


def get_data_loaders(config, dataset):
    """
    get data loaders for train and validation

    Args:
        config (config node object)
    Returns:
        train_loader (torch.utils.data.DataLoader)
        valid_loader (torch.utils.data.DataLoader)
    """
    dataloader = DataLoader(dataset=dataset,
                              batch_size=config.data.batch_size,
                              shuffle=config.data.shuffle, sampler=None,
                              batch_sampler=None, num_workers=0,
                              pin_memory=config.data.pin_memory,
                              drop_last=config.data.drop_last)




    return dataloader
