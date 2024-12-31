import argparse
import os
import shutil

import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    return device


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def check_dir(args):
    ensure_dir_exists('./result')
    ensure_dir_exists(args.model_path)
    ensure_dir_exists(args.log_path)
    if args.resume:
        ensure_dir_exists(args.visual_path)
    else:
        remove_and_create_dir(args.visual_path)


def get_model(args):
    print('★' * 30)
    print(f'project_name: {args.project_name}, model: {args.arch}')
    print('★' * 30)
    device = get_device()
    if args.project_name == 'DCNN':
        if args.arch == 'unet':
            from DCNN.unet import UNet
            model = UNet(in_ch=4, out_ch=1, list_ch=[-1, 32, 64, 128, 256]).to(device)
            return model

        elif 'efficientnet' in args.arch:
            from DCNN.efficientnet_unet import EfficientUNet
            model = EfficientUNet(in_chans=4, num_classes=1, pretrain_backbone=True, model_name=args.arch).to(
                device)
            return model
        elif args.arch == 'resunet':
            from DCNN.resunet import ResUNet
            model = ResUNet(in_chans=4, num_classes=1, pretrain_backbone=True).to(device)
            return model

    elif args.project_name == 'C3D':
        in_channel = 3
        if args.arch == 'unet':
            from C3D.model import Model
            model = Model(in_ch=in_channel, out_ch=1,
                          list_ch_A=[-1, 16, 32, 64, 128, 256],
                          list_ch_B=[-1, 32, 64, 128, 256, 512]).to(device)
            return model
        elif args.arch == 'resunet':
            from C3D.resunet import ResUNet
            model = ResUNet(in_channel).to('cuda')
            return model
        elif args.arch == 'cascade_resunet':
            from C3D.cascade_resunet import CascadeResUNet
            model = CascadeResUNet(in_channel=in_channel).to("cuda")
            return model
    else:
        raise ValueError('arch error')


def parse_args():
    from utils import set_seed
    set_seed(3407)

    parser = argparse.ArgumentParser(description="Train or test the dose prediction model")
    parser.add_argument('--project_name', type=str, default='C3D', help="project name")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet_b1', help='unet/efficientnet_b1')
    parser.add_argument("--volume_size", type=tuple, default=[128, 128, 128], help='volume size')
    # parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--resume', default=False, type=bool, help="Resume from the last checkpoint")
    parser.add_argument('--TTA', type=bool, default=False, help='do test-time augmentation, default True')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
