# -*- encoding: utf-8 -*-
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, predictions, targets):
        pred_A, pred_B = predictions
        gt_dose, possible_dose_mask = targets

        pred_A = pred_A[possible_dose_mask > 0]
        pred_B = pred_B[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        a = 0.3
        loss = a * self.l1_loss(pred_A, gt_dose) + (1 - a) * self.l1_loss(pred_B, gt_dose)
        return loss


class Loss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, predictions, targets):
        predicted_dose = predictions[0]
        gt_dose, possible_dose_mask = targets

        pred = predicted_dose[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        loss = self.l1_loss(pred, gt_dose)
        return loss
