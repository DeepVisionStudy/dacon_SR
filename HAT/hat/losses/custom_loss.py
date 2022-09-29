import torch
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils.color_util import rgb2ycbcr_pt

import numpy as np

def psnr_loss(pred, target):
    return -torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10. / np.log(10)
        self.toY = toY

    def forward(self, pred, target, **kwargs):
        pred = rgb2ycbcr_pt(pred, y_only=self.toY)
        target = rgb2ycbcr_pt(target, y_only=self.toY)

        return self.loss_weight * self.scale * psnr_loss(pred, target)