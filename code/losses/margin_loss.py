"""
Margin loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


def margin_loss(input_, target, pos_margin=0.9, neg_margin=0.1,
                pos_weight=1.0, neg_weight=0.5, reduction='mean',
                from_logits=False, epsilon=1e-7):
    """
    Margin loss
    """
    if not input_.shape == target.shape:
        raise ValueError

    if not from_logits:
        input_ = torch.clamp(input_, min=epsilon, max=(1 - epsilon))
        input_ = torch.log(input_ / (1 - input_))

    if not (input_.max() <= 1.0 and input_.min() >= 0.0):
        raise ValueError

    if not ((target.max() == 1.0 and target.min() == 0.0 and(target.unique().numel() == 2)) 
        or (target.max() == 0.0 and target.min() == 0.0 and(target.unique().numel() == 1))):
        raise ValueError

    pos_mask = target * (input_ < pos_margin)
    neg_mask = (1 - target) * (input_ > neg_margin)

    pos_loss = pos_mask.float() * (input_ - pos_margin) ** 2
    neg_loss = neg_mask.float() * (input_ - neg_margin) ** 2

    loss = pos_weight * pos_loss + neg_weight * neg_loss

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError

    return loss


class MarginLoss(nn.Module):
    """
    Margin loss module
    """

    def __init__(self, pos_margin=0.9, neg_margin=0.1, pos_weight=1.0,
                 neg_weight=0.5, reduction='mean', from_logits=False):
        super(MarginLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, input_, target):
        loss = margin_loss(input_=input_, target=target,
                           pos_margin=self.pos_margin,
                           neg_margin=self.neg_margin,
                           pos_weight=self.pos_weight,
                           neg_weight=self.neg_weight,
                           reduction=self.reduction,
                           from_logits=self.from_logits)
        return loss
