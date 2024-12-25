# -*- encoding: utf-8 -*-
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, predictions, targets):
        predicted_dose = predictions[0]
        gt_dose, possible_dose_mask = targets

        pred_dose = predicted_dose[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        loss = self.l1_loss(pred_dose, gt_dose)
        return loss
