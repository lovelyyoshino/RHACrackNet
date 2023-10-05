"""
Helper function for get optimizer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging


def get_optimizer(config, params):
    """
    Return a model object

    Args:
        config (config node object): config

    Returns:
        optimizer (torch.optim.Optimizer object): pytorch optimizer
    """

    if config.optimizer.optimizer_name == 'sgd':
        from torch.optim import SGD
        optimizer = SGD(params=params,
                        lr=config.optimizer.lr,
                        momentum=config.optimizer.momentum,
                        dampening=config.optimizer.dampening,
                        weight_decay=config.optimizer.weight_decay,
                        nesterov=config.optimizer.use_nesterov)

    elif config.optimizer.optimizer_name == 'adam':
        from torch.optim import Adam
        optimizer = Adam(params=params,
                         lr=config.optimizer.lr,
                         betas=(0.9, 0.999),
                         eps=1e-8, weight_decay=0,
                         amsgrad=False)
    else:
        logging.getLogger('Get Optimizer').error(
            'Optimizer for %s not implemented', config.optimizer.optimizer_name)
        raise NotImplementedError

    return optimizer
