# -*- encoding: utf-8 -*-
import os

from dataloader_OpenKBP_DCNN import get_train_loader
from loss import Loss
from network_trainer import NetworkTrainer
from online_evaluation import online_evaluation
from parse_args import get_model, parse_args

if __name__ == '__main__':
    args = parse_args()
    #  Start training
    trainer = NetworkTrainer(args)

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    trainer.setting.train_loader = get_train_loader(batch_size=args.batch_size, num_workers=num_workers)

    trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr_encoder': 3e-4,
                              'lr_decoder': 3e-4,
                              'weight_decay': 1e-4
                          }
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='cosine',
                             args={
                                 'T_max': args.epochs,
                                 'eta_min': 1e-7,
                                 'last_epoch': -1
                             }
                             )
    if args.resume:
        trainer.init_trainer(ckpt_file=trainer.setting.latest_ckpt_file,
                             only_network=False)

    trainer.run()
