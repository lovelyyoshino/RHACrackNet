"""
Dice coefficient
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


def soft_dice(input_, target, loss_type='jaccard', reduction='mean',
              from_logits=False, epsilon=1e-7):
    """
    Soft (differentiable) dice score coefficient
    """
    if not input_.shape == target.shape:
        raise ValueError

    if not from_logits:
        input_ = torch.clamp(input_, min=epsilon, max=(1 - epsilon))
        input_ = torch.log(input_ / (1 - input_))

    if loss_type == 'jaccard':
        input_norm = torch.sum(input_ * input_, dim=-1)
        target_norm = torch.sum(target * target, dim=-1)
    elif loss_type == 'sorensen':
        input_norm = torch.sum(input_, dim=-1)
        target_norm = torch.sum(target, dim=-1)
    else:
        raise ValueError

    intesection = torch.sum(input_ * target, dim=-1)
    dice = torch.div(2.0 * intesection + epsilon,
                     input_norm + target_norm + epsilon)

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        dice = torch.mean(dice)
    elif reduction == 'sum':
        dice = torch.sum(dice)
    else:
        raise NotImplementedError

    return dice


class SoftDice(nn.Module):
    """
    Soft (differentiable) dice score coefficient
    """

    def __init__(self, loss_type='jaccard', reduction='mean',
                 from_logits=False, epsilon=1e-7):
        super(SoftDice, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.from_logits = from_logits
        self.epsilon = epsilon

    def forward(self, input_, target):
        dice = soft_dice(input_=input_, target=target,
                         loss_type=self.loss_type,
                         reduction=self.reduction,
                         from_logits=self.from_logits,
                         epsilon=self.epsilon)

        return dice


def dice_loss(input_, target, from_logits=False, loss_type='jaccard',
              reduction='mean'):
    loss = 1 - soft_dice(input_=input_, target=target, loss_type=loss_type,
                         reduction=reduction, from_logits=from_logits)
    return loss


class DiceLoss(nn.Module):
    def __init__(self, loss_type='jaccard', from_logits=False,
                 reduction='mean'):
        super(DiceLoss, self).__init__()
        self.loss_type = loss_type
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, input_, target):
        loss = dice_loss(input_, target, from_logits=self.from_logits,
                         loss_type=self.loss_type, reduction=self.reduction)
        return loss

