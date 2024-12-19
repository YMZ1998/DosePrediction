# -*- encoding: utf-8 -*-
import argparse
import os

from dataloader_OpenKBP_C3D import get_train_loader
from loss import Loss
from model import Model
from network_trainer import NetworkTrainer
from online_evaluation import online_evaluation

# if os.path.abspath('..') not in sys.path:
#     sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[0], help='list_GPU_ids for training')
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--resume', default=True, type=bool, help="Resume from the last checkpoint")
    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'C3D'
    trainer.setting.output_dir = '../Output/C3D'
    list_GPU_ids = args.list_GPU_ids

    trainer.setting.network = Model(in_ch=9, out_ch=1,
                                    list_ch_A=[-1, 16, 32, 64, 128, 256],
                                    list_ch_B=[-1, 32, 64, 128, 256, 512])

    trainer.setting.max_epoch = args.epochs

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    trainer.setting.train_loader = get_train_loader(batch_size=args.batch_size, num_workers=num_workers)

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr': 3e-4,
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
        trainer.init_trainer(ckpt_file='../Output/C3D/latest.pkl',
                             list_GPU_ids=list_GPU_ids,
                             only_network=False)
    else:
        trainer.set_GPU_device(list_GPU_ids)

    os.makedirs(trainer.setting.output_dir, exist_ok=True)
    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')
