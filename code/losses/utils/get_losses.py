"""
Helper functions for get module or functional losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
sys.path.append('../../')




def get_loss_module(config):
    """
    Return a nn.Module object of loss

    Args:
        config (config node object)
    Returns:
        loss (torch.nn.Module): a loss module
    """
    if config.loss.loss_name == 'mse':
        from torch.nn import MSELoss
        loss = MSELoss(reduction=config.loss.reduction)

    elif config.loss.loss_name == 'wbce':
        from losses.cross_entropy import WeightedBCELoss
        from dataset.utils.get_statistics import get_pos_weight
        pos_weight = get_pos_weight(config)

        loss = WeightedBCELoss(
            pos_weight=pos_weight,
            weight=None,
            reduction=config.loss.reduction,
            from_logits=False,
        )

    elif config.loss.loss_name == 'bce':
        from torch.nn import BCELoss

        loss = BCELoss(weight=None, size_average=None,
                       reduce=None, reduction=config.loss.reduction)

    elif config.loss.loss_name == 'dice_loss':
        from losses.dice_coefficient import DiceLoss
        loss = DiceLoss(
            loss_type='jaccard',
            from_logits=False,
            reduction=config.loss.reduction,
        )

    elif config.loss.loss_name == 'dma_combine_loss':
        from losses.combine_loss import CombinedLoss
        from dataset.utils.get_statistics import get_pos_weight
        pos_weight = get_pos_weight(config)
        loss = CombinedLoss(
            pos_weight=pos_weight,
            weight=None,
            reduction=config.loss.reduction,
            from_logits=False,
            loss_type='jaccard',
        )

    elif config.loss.loss_name == 'jaccard':
        from losses.dice_coefficient import SoftDice
        loss = SoftDice(
            loss_type='jaccard',
            reduction=config.loss.reduction,
            from_logits=False,
            epsilon=1e-7
        )

    elif config.loss.loss_name == 'sorensen':
        from losses.dice_coefficient import SoftDice
        loss = SoftDice(
            loss_type='sorensen',
            reduction=config.loss.reduction,
            from_logits=False,
            epsilon=1e-7
        )

    elif config.loss.loss_name == 'focalloss_Sig':
        from losses.cross_entropy import FocalLossForSigmoid
        loss0 = FocalLossForSigmoid(
            gamma=config.loss.gamma,
            alpha=config.loss.alpha,
            reduction=config.loss.reduction
        )
        return loss0
    else:
        logging.getLogger('Get Loss Module').error(
            'Loss module for %s not implemented', config.loss.loss_name)
        raise NotImplementedError

    return loss


def get_loss_functional(config):
    """
    Return a callable object of loss

    Args:
        config (config node object): config
    Returns:
        loss (torch.nn.Module): a loss module
    """
    if config.loss.loss_name == 'mse':
        from torch.nn.functional import mse_loss
        loss_fn = mse_loss

    elif config.loss.loss_name == 'WBCE':
        from losses.cross_entropy import weighted_binary_cross_entropy
        loss_fn = weighted_binary_cross_entropy

    elif config.loss.loss_name == 'dice_loss':
        from losses.dice_coefficient import dice_loss
        loss_fn = dice_loss

    elif config.loss.loss_name == 'jaccard' or 'sorensen':
        from losses.dice_coefficient import soft_dice
        from functools import partial
        loss_fn = partial(soft_dice, loss_type=config.loss.loss_name)

    elif config.loss.loss_name == 'margin':
        from losses.margin_loss import margin_loss
        loss_fn = margin_loss

    else:
        logging.getLogger('Get Loss Functional').error(
            'Loss module for %s not implemented', config.loss.loss_name)
        raise NotImplementedError

    return loss_fn
