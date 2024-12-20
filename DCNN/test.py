# -*- encoding: utf-8 -*-
import argparse
import os
import shutil
import sys

import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm

from dataloader_OpenKBP_DCNN import read_data, pre_processing
from evaluate_openKBP import get_Dose_score_and_DVH_score
from model import Model
from network_trainer import NetworkTrainer
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
        [prediction] = trainer.setting.network(augmented_input)

        # Aug back to original order
        prediction = flip_2d(np.array(prediction.cpu().data[0, :, :, :]), list_flip_axes)

        list_predictions.append(prediction[0, :, :])

    return np.mean(list_predictions, axis=0)


def inference(trainer, list_patient_dirs, save_path, do_TTA=True):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--TTA', type=bool, default=False, help='do test-time augmentation, default True')
    parser.add_argument('--GPU_id', type=int, default=0, help='GPU_id for testing (default: 0)')
    args = parser.parse_args()

    trainer = NetworkTrainer('DCNN')

    trainer.setting.network = Model(in_ch=4, out_ch=1,
                                    list_ch=[-1, 32, 64, 128, 256])

    # Load model weights
    trainer.init_trainer(ckpt_file=trainer.setting.best_ckpt_file,
                         list_GPU_ids=[args.GPU_id],
                         only_network=True)

    save_path = os.path.join(trainer.setting.output_dir, 'Prediction_' + str(args.TTA))

    # Start inference
    print('Start inference !')
    list_patient_dirs = ['../Data/OpenKBP_DCNN/pt_' + str(i) for i in range(201, 241)]
    inference(trainer, list_patient_dirs, save_path=save_path, do_TTA=args.TTA)

    # Evaluation
    print('Start evaluation !')
    Dose_score, DVH_score = get_Dose_score_and_DVH_score(prediction_dir=save_path, gt_dir='../Data/OpenKBP_C3D')
    print('TTA :', args.TTA)
    print('Dose score is: ' + str(Dose_score))
    print('DVH score is: ' + str(DVH_score))
