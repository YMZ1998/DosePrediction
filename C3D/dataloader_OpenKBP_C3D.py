# -*- encoding: utf-8 -*-
import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2
from torch.utils.data import DataLoader

from DataAugmentation.augmentation_OpenKBP_C3D import \
    random_flip_3d, random_rotate_around_z_axis, random_translate, to_tensor

"""
images are always C*Z*H*W
"""


def read_data(patient_dir):
    if not os.path.exists(patient_dir):
        raise ValueError('Patient directory does not exist!')
    dict_images = {}
    list_structures = ['CT',
                       'PTV70',
                       'PTV63',
                       'PTV56',
                       'possible_dose_mask',
                       'Brainstem',
                       'SpinalCord',
                       'RightParotid',
                       'LeftParotid',
                       'Esophagus',
                       'Larynx',
                       'Mandible',
                       'dose']

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '.nii.gz'

        if structure_name == 'CT':
            dtype = sitk.sitkInt16
        elif structure_name == 'dose':
            dtype = sitk.sitkFloat32
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            # To numpy array (C * Z * H * W)
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)
    # print(patient_dir, np.sum(dict_images['possible_dose_mask']))
    return dict_images


def img_normalize(img):
    min_value = np.min(img)
    max_value = np.max(img)
    img = (img - min_value) / (max_value - min_value + 1e-8)
    # img = img * 2 - 1
    return img


def pre_processing(dict_images):
    # PTVs
    PTVs = 70.0 / 70. * dict_images['PTV70'] \
           + 63.0 / 70. * dict_images['PTV63'] \
           + 56.0 / 70. * dict_images['PTV56']

    # OARs
    list_OAR_names = ['Brainstem',
                      'SpinalCord',
                      'RightParotid',
                      'LeftParotid',
                      'Esophagus',
                      'Larynx',
                      'Mandible']
    OAR_all = np.concatenate([dict_images[OAR_name] for OAR_name in list_OAR_names], axis=0)
    # oar = np.zeros_like(PTVs, np.uint8)
    # for i in range(7):
    #     oar += dict_images[list_OAR_names[i]]

    # OAR_all = np.zeros((1, 128, 128, 128), np.uint8)
    # for OAR_i in range(7):
    #     OAR = dict_images[list_OAR_names[OAR_i]]
    #     OAR_all[OAR > 0] = 1
        # OAR_all[OAR > 0] = OAR_i + 1
    # print(oar,OAR_all)
    # print(PTVs.shape)
    # print(OAR_all.shape, OAR_all.max(), OAR_all.min())
    # print(oar.shape, oar.max(), oar.min())

    # CT image
    CT = dict_images['CT'].astype(np.float32)
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    # CT = CT.astype(np.float32) / 1000.
    CT = img_normalize(CT)

    # Dose image
    dose = dict_images['dose'] / 70.

    # Possible_dose_mask, the region that can receive dose
    possible_dose_mask = dict_images['possible_dose_mask']

    list_images = [np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
                   dose,  # Label
                   possible_dose_mask]
    return list_images


def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angles=(0, 40, 80, 120, 160, 200, 240, 280, 320),
                                              list_boder_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(list_images,
                                   roi_mask=list_images[2][0, :, :, :],  # the possible dose mask
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0])

    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


class MyDataset(data.Dataset):
    def __init__(self, phase):
        # 'train' or 'val
        self.phase = phase
        self.transform = {'train': train_transform, 'val': val_transform}

        self.list_case_id = {'train': ['../Data/OpenKBP_C3D/pt_' + str(i) for i in range(1, 241)],
                             'val': ['../Data/OpenKBP_C3D/pt_' + str(i) for i in range(201, 241)]}[phase]

        random.shuffle(self.list_case_id)

    def __getitem__(self, index):
        case_id = self.list_case_id[index]

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images)

        list_images = self.transform[self.phase](list_images)
        return list_images

    def __len__(self):
        return len(self.list_case_id)


def get_train_loader(batch_size=1, num_workers=8):
    train_dataset = MyDataset(phase='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=False)
    return train_loader


def get_val_loader(batch_size=1, num_workers=8):
    val_dataset = MyDataset(phase='val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=False)
    return val_loader


if __name__ == '__main__':
    train_loader = get_train_loader(batch_size=1, num_workers=8)
    for i, data in enumerate(train_loader):
        print(i, data[0].shape, data[1].shape, data[2].shape)
        print(data[0].max(), data[0].min())
