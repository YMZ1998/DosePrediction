import argparse
import os
import shutil

import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    return device


# def get_best_weight_path(args, verbose=False):
#     weights_path = "./{}_{}_best_model.pth".format(args.arch, args.anatomy)
#     if verbose:
#         print("best weight: ", weights_path)
#     return weights_path
#
#
# def get_latest_weight_path(args, verbose=False):
#     weights_path = "checkpoint/{}_{}_latest_model.pth".format(args.arch, args.anatomy)
#     if verbose:
#         print("latest weight: ", weights_path)
#     return weights_path


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
    print(f'model:{args.arch}')
    print('★' * 30)
    device = get_device()
    if args.arch == 'unet':
        from unet import UNet
        model = UNet(in_ch=4, out_ch=1, list_ch=[-1, 32, 64, 128, 256]).to(device)
        return model

    elif 'efficientnet' in args.arch:
        from efficientnet_unet import EfficientUNet
        model = EfficientUNet(in_chans=4, num_classes=1, pretrain_backbone=False, model_name='efficientnet_b0').to(
            device)
        return model

    else:
        raise ValueError('arch error')


def parse_args():
    from utils import set_seed
    set_seed(3407)

    parser = argparse.ArgumentParser(description="Train or test the dose prediction model")
    parser.add_argument('--project_name', type=str, default='DCNN', help="project name")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='unet', help='unet/efficientnet_b0')
    # parser.add_argument("--image_size", default=128, type=int)
    # parser.add_argument('--learning_rate', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument("--epochs", default=400, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--resume', default=False, type=bool, help="Resume from the last checkpoint")
    parser.add_argument('--TTA', type=bool, default=False, help='do test-time augmentation, default True')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
