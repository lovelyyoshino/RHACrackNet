"""
Dice coefficient
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from losses.dice_coefficient import DiceLoss
from losses.cross_entropy import WeightedBCELoss

class CombinedLoss(nn.Module):
    def __init__(self, pos_weight, weight=None, reduction='mean',
                 from_logits=False,loss_type='jaccard'):
        super(CombinedLoss, self).__init__()
        self.pos_weight = pos_weight
        self.weight = weight
        self.reduction = reduction
        self.from_logits = from_logits
        self.loss_type = loss_type

        self.dice_loss = DiceLoss(
            loss_type=self.loss_type,
            from_logits=self.from_logits,
            reduction=self.reduction
        )

        self.bce_loss = WeightedBCELoss(
            pos_weight=self.pos_weight,
            weight=self.weight,
            reduction=self.reduction,
            from_logits=self.from_logits
        )

    def forward(self, input_, target):
        dice_loss = self.dice_loss(input_, target)
        bce_loss = self.bce_loss(input_, target)
        combined_loss = dice_loss + bce_loss
        return combined_loss
