"""
Helper functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging


def get_lr_schedule(config, optimizer):
    """
    Return a learning rate schedule object

    Args:
        config (config node object): config

    Returns:
        optimizer (torch.optim.Optimizer object): pytorch optimizer
    """

    if config.optimizer.lr_scheduler.lr_scheduler_name == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode='min',
            factor=config.optimizer.lr_scheduler.factor,
            patience=config.optimizer.lr_scheduler.patience,
            min_lr=config.optimizer.lr_scheduler.min_lr,
            verbose=True, threshold=0.0001, threshold_mode='rel',
            cooldown=0, eps=1e-08,
        )

    elif config.optimizer.lr_scheduler.lr_scheduler_name == 'cyclic':
        from torch.optim.lr_scheduler import CyclicLR
        scheduler = CyclicLR(
            optimizer=optimizer,
            base_lr=config.optimizer.lr_scheduler.base_lr,
            max_lr=config.optimizer.lr_scheduler.max_lr,
            step_size_up=2000,
            step_size_down=None,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle',
            cycle_momentum=True,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=-1
        )

    elif config.optimizer.lr_scheduler.lr_scheduler_name == 'CosAnnealing':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.optimizer.lr_scheduler.T_max,
            eta_min = config.optimizer.lr_scheduler.min_lr,
            last_epoch=-1
        )

    else:
        logging.getLogger('Get LR Schedule').error(
            'Schedule for %s not implemented',
            config.lr_scheduler.lr_scheduler_name)
        raise NotImplementedError

    return scheduler
