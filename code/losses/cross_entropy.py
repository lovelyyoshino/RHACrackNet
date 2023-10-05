"""
Cross entropy loss for imbalanced problem
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


def weighted_binary_cross_entropy(input_, target, pos_weight, weight=None,
                                  reduction='mean', from_logits=False,
                                  epsilon=1e-7):
    """
    Weighted binary cross entropy
    """
    if not input_.shape == target.shape:
        raise ValueError(input_.shape, target.shape)

    if not from_logits:
        input_ = torch.clamp(input_, min=epsilon, max=(1 - epsilon))
        input_ = torch.log(input_ / (1 - input_))

    max_val = torch.clamp(-1 * input_, min=0)
    balanced_weight = 1 + target * (pos_weight - 1)
    loss = (1 - target) * input_ + balanced_weight * (torch.log(
        torch.exp(-1 * max_val) + torch.exp(-1 * input_ - max_val)
    ) + max_val)

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError

    return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross entropy module
    """

    def __init__(self, pos_weight, weight=None, reduction='mean',
                 from_logits=False):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.weight = weight
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, input_, target):
        loss = weighted_binary_cross_entropy(
            input_=input_, target=target,
            pos_weight=self.pos_weight,
            weight=self.weight,
            reduction=self.reduction,
            from_logits=self.from_logits
        )
        return loss


class FocalLossForSigmoid(nn.Module):
    def __init__(self, gamma=3, alpha=None, reduction='mean'):
        super(FocalLossForSigmoid, self).__init__()
        self.gamma = gamma
        assert 0 <= alpha <= 1, 'The value of alpha must in [0,1]'
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCELoss(reduce=False)

    def forward(self, input_, target):
        input_ = torch.clamp(input_, min=1e-7, max=(1 - 1e-7))

        if self.alpha != None:
            loss = (self.alpha * target + (1 - target) * (1 - self.alpha)) * (
                torch.pow(torch.abs(target - input_), self.gamma)) * self.bce(input_, target)
        else:
            loss = torch.pow(torch.abs(target - input_), self.gamma) * self.bce(input_, target)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            pass

        return loss
