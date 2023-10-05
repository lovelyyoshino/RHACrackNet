"""
Helper functions for calculating statistics of datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from PIL import Image
from torchvision.transforms import functional as TF
#import sys
#sys.path.append('/home/BenzhangQiu/pytorch/project/Crack Detection/dataset/utils/')
from .get_datasets import get_datasets


def load_pos_weight(config):
    """
    Load calculated positive weights.
    """
    if config.data.dataset_name == 'chase':
        stat_dict = {
            'mean': [0.45290204882621765, 0.16392576694488525, 0.02793888747692108],
            'std': [0.3399156928062439, 0.14024977385997772, 0.0352049358189106],
            'positive': 1861974,
            'negative': 24991146,
        }

    elif config.data.dataset_name == 'drive':
        # pre-calculated mean and std of first 20 sample (official train set)
        # stat_dict = {
        #     'mean': [0.4973878860473633, 0.27064797282218933, 0.16244100034236908],
        #     'std': [0.3316500782966614, 0.17835408449172974, 0.09872566163539886],
        #     'positive': 569615,
        #     'negative': 6029585,
        # }

        stat_dict = {
            'mean': [0.5078321695327759, 0.26819586753845215, 0.16155381500720978],
            'std': [0.33780568838119507, 0.17534150183200836, 0.09775665402412415],
            'positive': 1147560,
            'negative': 12050840,
        }

    elif config.data.dataset_name == 'hrf':
        stat_dict = {
            'mean': [0.620665967464447, 0.2001914530992508, 0.1024790033698082],
            'std': [0.28755491971969604, 0.10416200757026672, 0.054919030517339706],
            'positive': 28397335,
            'negative': 339943145,
        }

    elif config.data.dataset_name == 'stare':
        stat_dict = {
            'mean': [0.5889195799827576, 0.3338136374950409, 0.11336859315633774],
            'std': [0.3377639353275299, 0.17784464359283447, 0.06998658925294876],
            'positive': 644053,
            'negative': 7825947,
        }

    else:
        logging.getLogger('Load Pos Weight').warnning(
            'Dataset statistic for %s is not specified, calculating now...',
            config.data.dataset_name)
        return get_pos_weight(config)

    pos = stat_dict['positive']
    neg = stat_dict['negative']
    pos_weight = (pos + neg) / pos * config.loss.pos_weight_factor
    return pos_weight


def get_pos_weight(config):
    """
    Get pos_weight argument for loss functions, where
         pos_weight = (pos + neg) / pos * pos_weight_factor
    """
    *_, pos, neg = get_dataset_statistics(config)
    pos_weight = (pos + neg) / pos * config.loss.pos_weight_factor
    return pos_weight


def get_dataset_statistics(config):
    """
    Get statistics of train dataset, including channel-wise mean and std,
         positive pixel count, and negative pixel count
    """
    dataset,_ = get_datasets(config.data.dataset_train_name, config.data.data_root)

    mean, std = 0.0, 0.0
    pos_count, neg_count = 0, 0
    for image, annot, *_ in dataset:
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)

        num_channel, *_ = image.shape
        mean += image.view(num_channel, -1).mean(-1)
        std += image.view(num_channel, -1).std(-1)

        if isinstance(annot, Image.Image):
            annot = TF.to_tensor(annot)

        assert annot.min() == 0 and annot.max() == 1
        assert len(annot.unique()) == 2
        pos_count += annot.sum().numpy()
        neg_count += (annot.numel() - annot.sum()).numpy()

    mean /= len(dataset)
    std /= len(dataset)
    mean = mean.numpy().tolist()
    std = std.numpy().tolist()

    return mean, std, pos_count, neg_count
