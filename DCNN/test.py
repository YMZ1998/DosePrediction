# -*- encoding: utf-8 -*-
import os
import sys
import time

import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm

from dataloader_OpenKBP_DCNN import read_data, pre_processing
from evaluate_openKBP import evaluate_OpenKBP
from network_trainer import NetworkTrainer
from parse_args import parse_args, remove_and_create_dir
from utils import copy_image_info


# Input is C*H*W
def flip_2d(input_, list_axes):
    if 'W' in list_axes:
        input_ = input_[:, :, ::-1]

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    list_predictions = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_2d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).to(trainer.setting.device)
        prediction = trainer.setting.network(augmented_input)

        # Aug back to original order
        prediction = flip_2d(np.array(prediction.cpu().data[0, :, :, :]), list_flip_axes)

        list_predictions.append(prediction[0, :, :])

    return np.mean(list_predictions, axis=0)


def inference(trainer, list_patient_dirs, save_path, do_TTA=True):
    remove_and_create_dir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in tqdm(list_patient_dirs, file=sys.stdout):
            patient_id = patient_dir.split('/')[-1]

            prediction_dose = np.zeros((128, 128, 128), np.float32)
            gt_dose = np.zeros((128, 128, 128), np.float32)
            possible_dose_mask = np.zeros((128, 128, 128), np.uint8)

            for slice_i in range(128):
                if not os.path.exists(patient_dir + '/CT_' + str(slice_i) + '.nii.gz'):
                    continue

                # Read data and pre-process
                dict_images = read_data(patient_dir, slice_i)
                list_images = pre_processing(dict_images)

                # Test-time augmentation
                if do_TTA:
                    TTA_mode = [[], ['W']]
                else:
                    TTA_mode = [[]]

                prediction_single_slice = test_time_augmentation(trainer, input_=list_images[0],
                                                                 TTA_mode=TTA_mode)
                prediction_dose[slice_i, :, :] = prediction_single_slice
                gt_dose[slice_i, :, :] = list_images[1][0, :, :]
                possible_dose_mask[slice_i, :, :] = list_images[2][0, :, :]

            # Pose-processing
            prediction_dose[np.logical_or(possible_dose_mask < 1, prediction_dose < 0)] = 0
            prediction_dose = 70. * prediction_dose

            # Save prediction to nii image
            templete_nii = sitk.ReadImage('../Data/OpenKBP_C3D/pt_1/possible_dose_mask.nii.gz')
            prediction_nii = sitk.GetImageFromArray(prediction_dose)
            prediction_nii = copy_image_info(templete_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + patient_id):
                os.mkdir(save_path + '/' + patient_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + patient_id + '/dose.nii.gz')


if __name__ == "__main__":
    if not os.path.exists('../Data/OpenKBP_C3D'):
        raise Exception('OpenKBP_C3D should be prepared before testing, please run prepare_OpenKBP_C3D.py')

    args = parse_args()
    args.project_name = 'DCNN'
    args.arch = 'efficientnet_b1'
    args.TTA = False

    trainer = NetworkTrainer(args)
    print(time.strftime('Local time: %H:%M:%S', time.localtime(time.time())))

    trainer.init_trainer(ckpt_file=trainer.setting.best_ckpt_file, only_network=True)

    save_path = os.path.join(trainer.setting.output_dir, 'Prediction_' + str(args.TTA))

    # Start inference
    print('Start inference !')
    print('Prediction will be saved to {}'.format(save_path))

    list_patient_dirs = ['../Data/OpenKBP_DCNN/pt_' + str(i) for i in range(201, 241)]
    # list_patient_dirs = ['../Data/OpenKBP_C3D/pt_' + str(i) for i in range(241, 341)]
    inference(trainer, list_patient_dirs, save_path=save_path, do_TTA=args.TTA)

    evaluate_OpenKBP(save_path)
