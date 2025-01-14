# -*- encoding: utf-8 -*-
import datetime
import os
import time

from dataloader_OpenKBP_C3D import get_train_loader
from loss import Loss, Loss2
from network_trainer import NetworkTrainer
from online_evaluation import online_evaluation
from parse_args import parse_args


def train_c3d():
    args = parse_args()
    args.project_name = 'C3D'
    args.batch_size = 2
    args.arch = 'cascade_resunet'
    # args.resume = True

    trainer = NetworkTrainer(args)

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    trainer.setting.train_loader = get_train_loader(batch_size=args.batch_size, num_workers=num_workers)
    if args.arch == 'resunet':
        trainer.setting.loss_function = Loss2()
    else:
        trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='AdamW', args={'lr': 5e-4})

    trainer.set_lr_scheduler(lr_scheduler_type='cosine', args={'T_max': args.epochs})
    if args.resume:
        trainer.init_trainer(ckpt_file=trainer.setting.latest_ckpt_file,
                             only_network=False)

    start_time = time.time()
    trainer.run()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    train_c3d()
