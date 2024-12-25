# -*- encoding: utf-8 -*-
import torch
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
        # print(torch.sum(possible_dose_mask))
        # print(pred_A.shape)
        if torch.sum(possible_dose_mask) < 0:
            raise ValueError('possible_dose_mask is all zero')
        pred_A = pred_A[possible_dose_mask > 0]
        pred_B = pred_B[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        a = 0.3
        L1_loss = a * self.L1_loss_func(pred_A, gt_dose) + (1-a) * self.L1_loss_func(pred_B, gt_dose)
        return L1_loss


class Loss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt):
        pred_A = pred[0]
        gt_dose = gt[0]
        possible_dose_mask = gt[1]
        # print(torch.sum(possible_dose_mask))
        # print(pred_A.shape)
        if torch.sum(possible_dose_mask) < 0:
            raise ValueError('possible_dose_mask is all zero')
        pred_A = pred_A[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        L1_loss = self.L1_loss_func(pred_A, gt_dose)
        return L1_loss
