# -*- encoding: utf-8 -*-
import datetime
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from DCNN.parse_args import get_model, get_device


class TrainerSetting:
    def __init__(self, args):
        self.project_name = args.project_name

        # Path for saving model and training log
        self.output_dir = os.path.join('../Output', self.project_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir,
                                     "log_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M")))
        self.latest_ckpt_file = os.path.join(self.output_dir, '{}_latest_model.pth'.format(args.arch))
        self.best_ckpt_file = os.path.join(self.output_dir, '{}_best_model.pth'.format(args.arch))

        # Generally only use one of them
        self.max_epoch = args.epochs
        self.network = get_model(args)
        self.device = get_device()

        self.train_loader = None
        self.val_loader = None

        self.optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_type = None

        self.loss_function = None

        # If do online evaluation during validation
        self.online_evaluation_function_val = None


class TrainerLog:
    def __init__(self):
        self.iter = 0
        self.epoch = 0

        # Moving average loss, loss is the smaller the better
        self.moving_train_loss = None
        # Average train loss of a epoch
        self.average_train_loss = 99999999.
        self.best_average_train_loss = 99999999.
        # Evaluation index is the lower the better
        self.average_val_index = 99999999.
        self.best_average_val_index = 99999999.

        self.save_status = []


class NetworkTrainer:
    def __init__(self, args):
        self.log = TrainerLog()
        self.setting = TrainerSetting(args)
    def set_optimizer(self, optimizer_type, args):
        # Sometimes we need set different learning rates for "encoder" and "decoder" separately
        if optimizer_type == 'Adam':
            if hasattr(self.setting.network, 'decoder') and hasattr(self.setting.network, 'encoder'):
                self.setting.optimizer = optim.Adam([
                    {'params': self.setting.network.encoder.parameters(), 'lr': args['lr_encoder']},
                    {'params': self.setting.network.decoder.parameters(), 'lr': args['lr_decoder']}
                ],
                    weight_decay=args['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    amsgrad=True)
            else:
                self.setting.optimizer = optim.Adam(self.setting.network.parameters(),
                                                    lr=args['lr'],
                                                    weight_decay=3e-5,
                                                    betas=(0.9, 0.999),
                                                    eps=1e-08,
                                                    amsgrad=True)

    def set_lr_scheduler(self, lr_scheduler_type, args):
        if lr_scheduler_type == 'step':
            self.setting.lr_scheduler_type = 'step'
            self.setting.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.setting.optimizer,
                                                                       milestones=args['milestones'],
                                                                       gamma=args['gamma'],
                                                                       last_epoch=args['last_epoch']
                                                                       )
        elif lr_scheduler_type == 'cosine':
            self.setting.lr_scheduler_type = 'cosine'
            # self.setting.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.setting.optimizer,
            #                                                                  T_max=args['T_max'],
            #                                                                  eta_min=args['eta_min'],
            #                                                                  last_epoch=args['last_epoch']
            #                                                                  )
            lf = lambda x: ((1 + math.cos(x * math.pi / args['T_max'])) / 2) * (1 - 0.01) + 0.01
            self.setting.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.setting.optimizer, lr_lambda=lf)
        elif lr_scheduler_type == 'ReduceLROnPlateau':
            self.setting.lr_scheduler_type = 'ReduceLROnPlateau'
            self.setting.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.setting.optimizer,
                                                                             mode='min',
                                                                             factor=args['factor'],
                                                                             patience=args['patience'],
                                                                             verbose=True,
                                                                             threshold=args['threshold'],
                                                                             threshold_mode='rel',
                                                                             cooldown=0,
                                                                             min_lr=0,
                                                                             eps=1e-08)

    def update_lr(self):
        # Update learning rate, only 'ReduceLROnPlateau' need use the moving train loss
        if self.setting.lr_scheduler_type == 'ReduceLROnPlateau':
            self.setting.lr_scheduler.step(self.log.moving_train_loss)
        else:
            self.setting.lr_scheduler.step()
        # lr = self.setting.optimizer.param_groups[0]["lr"]
        # print('Learning rate: %.7f' % lr)

    def update_moving_train_loss(self, loss):
        if self.log.moving_train_loss is None:
            self.log.moving_train_loss = loss.item()
        else:
            self.log.moving_train_loss = \
                (1 - self.setting.eps_train_loss) * self.log.moving_train_loss \
                + self.setting.eps_train_loss * loss.item()

    def update_average_statistics(self, loss):
        self.log.average_val_index = loss
        if loss < self.log.best_average_val_index:
            self.log.best_average_val_index = loss
            self.log.save_status.append('best')

    def forward(self, input_, phase):
        # To device
        input_ = input_.to(self.setting.device)

        # Forward
        if phase == 'train':
            self.setting.optimizer.zero_grad()
        output = self.setting.network(input_)

        return output

    def backward(self, output, target):
        for target_i in range(len(target)):
            target[target_i] = target[target_i].to(self.setting.device)

        # Optimize
        loss = self.setting.loss_function(output, target)
        loss.backward()
        self.setting.optimizer.step()

        return loss

    def train_one_epoch(self):
        self.setting.network.train()
        train_losses = []

        data_loader_train = tqdm(self.setting.train_loader, file=sys.stdout)
        for list_loader_output in data_loader_train:

            # List_loader_output[0] default as the input
            input_ = list_loader_output[0]
            target = list_loader_output[1:]

            output = self.forward(input_, phase='train')
            loss = self.backward(output, target)

            train_losses.append(loss.item())

            data_loader_train.desc = f"[train epoch {self.log.epoch}] loss: {np.mean(train_losses):.4f} "

        self.update_lr()

    def val(self):
        self.setting.network.eval()

        if self.setting.online_evaluation_function_val is None:
            self.print_log_to_file('==> No online evaluation method specified ! ')
            raise Exception('No online evaluation method specified !')
        else:
            val_index = self.setting.online_evaluation_function_val(self)
            self.update_average_statistics(val_index)

    def run(self):
        self.print_log_to_file('-' * 30)
        if self.log.iter == 0:
            self.print_log_to_file('Start training !', 'w')
        else:
            self.print_log_to_file('Continue training !', 'w')
        self.print_log_to_file(time.strftime('Local time: %H:%M:%S', time.localtime(time.time())))

        while (self.log.epoch < self.setting.max_epoch):
            self.print_log_to_file('-' * 30)
            time_start_this_epoch = time.time()
            self.log.epoch += 1
            self.print_log_to_file('Epoch {}/{}'.format(self.log.epoch, self.setting.max_epoch))
            self.print_log_to_file('Lr is %.6f, %.6f' % (
                self.setting.optimizer.param_groups[0]['lr'], self.setting.optimizer.param_groups[-1]['lr']))

            self.log.save_status = []

            self.train_one_epoch()
            if self.log.epoch % 3 == 0:
                self.val()

            self.log.save_status.append('latest')

            # Try save trainer
            if len(self.log.save_status) > 0:
                print('Saving trainer...')
                print(self.log.save_status)
                for status in self.log.save_status:
                    self.save_trainer(status=status)
                self.log.save_status = []

            self.print_log_to_file('Average val evaluation index is %.5f, best is %.5f'
                                   % (self.log.average_val_index, self.log.best_average_val_index))
            self.print_log_to_file('Total use time %.2f ' % (time.time() - time_start_this_epoch))
            self.print_log_to_file('-' * 30)

        self.print_log_to_file('End')

    def print_log_to_file(self, txt, mode='a'):
        with open(self.setting.log_file, mode) as log_:
            log_.write(txt + '\n')

        # Also display log in the terminal
        txt = txt.replace('\n', '')
        print(txt)

    def save_trainer(self, status='latest'):
        if len(self.setting.list_GPU_ids) > 1:
            network_state_dict = self.setting.network.module.state_dict()
        else:
            network_state_dict = self.setting.network.state_dict()

        optimizer_state_dict = self.setting.optimizer.state_dict()
        lr_scheduler_state_dict = self.setting.lr_scheduler.state_dict()

        ckpt = {
            'network_state_dict': network_state_dict,
            'lr_scheduler_state_dict': lr_scheduler_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'log': self.log
        }

        torch.save(ckpt, self.setting.output_dir + '/' + status + '.pkl')

    # Default load trainer in cpu, please reset device using the function self.set_GPU_device
    def init_trainer(self, ckpt_file, only_network=True):
        print('Loading ' + ckpt_file + '...')
        ckpt = torch.load(ckpt_file, weights_only=False, map_location='cpu')

        self.setting.network.load_state_dict(ckpt['network_state_dict'])

        if not only_network:
            self.setting.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            self.setting.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.log = ckpt['log']

        # If do not do so, the states of optimizer will always in cpu
        # This for Adam
        if type(self.setting.optimizer).__name__ == 'Adam':
            for key in self.setting.optimizer.state.items():
                key[1]['exp_avg'] = key[1]['exp_avg'].to(self.setting.device)
                key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(self.setting.device)
                key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(self.setting.device)
