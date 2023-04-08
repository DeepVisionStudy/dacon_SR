# https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/losses/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.losses.basic_loss import l1_loss

import numpy as np

def psnr_loss(pred, target):
    return torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10. / np.log(10)
        self.toY = toY

    def forward(self, pred, target):
        pred = rgb2ycbcr_pt(pred, y_only=self.toY)
        target = rgb2ycbcr_pt(target, y_only=self.toY)

        return self.loss_weight * self.scale * psnr_loss(pred, target)


@LOSS_REGISTRY.register()
class DownScaleLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(DownScaleLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        pred = F.interpolate(pred, size=(128, 128), mode='area')
        target = F.interpolate(target, size=(128, 128), mode='area')
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)