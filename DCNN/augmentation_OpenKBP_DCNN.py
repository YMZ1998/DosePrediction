# -*- encoding: utf-8 -*-
import numpy as np
import random
import cv2
import torch


# Random flip
def random_flip_2d(list_images, list_axis=(0, 1), p=0.5):
    if random.random() <= p:
        if 0 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, ::-1, :]
        if 1 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, ::-1]

    return list_images


# Random rotation using OpenCV
def random_rotate_around_z_axis(list_images,
                                list_angle,
                                list_interp,
                                list_boder_value,
                                p=0.5):
    if random.random() <= p:
        # Randomly pick an angle list_angles
        _angle = random.sample(list_angle, 1)[0]
        # Do not use random scaling, set scale factor to 1
        _scale = 1.

        for image_i in range(len(list_images)):
            for chan_i in range(list_images[image_i].shape[0]):
                rows, cols = list_images[image_i][chan_i, :, :].shape
                M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), _angle, scale=_scale)
                list_images[image_i][chan_i, :, :] = \
                    cv2.warpAffine(list_images[image_i][chan_i, :, :],
                                   M,
                                   (cols, rows),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=list_boder_value[image_i],
                                   flags=list_interp[image_i])
    return list_images


# Random translation
def random_translate(list_images, roi_mask, p, max_shift, list_pad_value):
    if random.random() <= p:
        exist_mask = np.where(roi_mask > 0)
        ori_h, ori_w = list_images[0].shape[1:]

        bh = min(max_shift-1, np.min(exist_mask[0]))
        eh = max(ori_h-1-max_shift, np.max(exist_mask[0]))
        bw = min(max_shift-1, np.min(exist_mask[1]))
        ew = max(ori_w-1-max_shift, np.max(exist_mask[1]))

        for image_i in range(len(list_images)):
            list_images[image_i] = list_images[image_i][:, bh:eh + 1, bw:ew + 1]

        list_images = random_pad_to_size_2d(list_images,
                                            target_size=[ori_h, ori_w],
                                            list_pad_value=list_pad_value)

    return list_images


# To tensor, images should be C*Z*H*W
def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images


# Pad
def random_pad_to_size_2d(list_images, target_size, list_pad_value):
    _, ori_h, ori_w = list_images[0].shape[:]
    new_h, new_w = target_size[:]

    pad_h = new_h - ori_h
    pad_w = new_w - ori_w

    pad_h_1 = random.randint(0, pad_h)
    pad_w_1 = random.randint(0, pad_w)

    pad_h_2 = pad_h - pad_h_1
    pad_w_2 = pad_w - pad_w_1

    output = []
    for image_i in range(len(list_images)):
        _image = list_images[image_i]
        output.append(np.pad(_image,
                             ((0, 0), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                             mode='constant',
                             constant_values=list_pad_value[image_i])
                      )
    return output
