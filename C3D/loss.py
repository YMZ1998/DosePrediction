# -*- encoding: utf-8 -*-
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt):
        pred_A = pred[0]
        pred_B = pred[1]
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        pred_A = pred_A[possible_dose_mask > 0]
        pred_B = pred_B[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        L1_loss = 0.5 * self.L1_loss_func(pred_A, gt_dose) + self.L1_loss_func(pred_B, gt_dose)
        return L1_loss
