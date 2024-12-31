import os
import sys

import SimpleITK as sitk
import numpy as np
import onnxruntime
from tqdm import tqdm

from dataloader_OpenKBP_C3D import read_data, pre_processing
from evaluate_openKBP import evaluate_OpenKBP
from network_trainer import NetworkTrainer
from utils import copy_image_info


def val_onnx(onnx_file_path, list_patient_dirs, save_path):
    remove_and_create_dir(save_path)

    for patient_dir in tqdm(list_patient_dirs, file=sys.stdout):
        patient_id = patient_dir.split('/')[-1]

        dict_images = read_data(patient_dir)
        list_images = pre_processing(dict_images)

        input = list_images[0]
        possible_dose_mask = list_images[-1]

        session = onnxruntime.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])

        input = np.expand_dims(input, 0).astype(np.float32)
        print(np.sum(input))
        output_name = session.get_outputs()[0].name
        ort_inputs = {session.get_inputs()[0].name: (input)}
        result = session.run([output_name], ort_inputs)
        prediction = np.squeeze(result)

        # Pose-processing
        prediction[np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)] = 0
        prediction = 70. * prediction

        # Save prediction to nii image
        templete_nii = sitk.ReadImage(patient_dir + '/CT.nii.gz')
        prediction_nii = sitk.GetImageFromArray(prediction)
        prediction_nii = copy_image_info(templete_nii, prediction_nii)
        if not os.path.exists(save_path + '/' + patient_id):
            os.mkdir(save_path + '/' + patient_id)
        sitk.WriteImage(prediction_nii, save_path + '/' + patient_id + '/dose.nii.gz')


if __name__ == '__main__':
    from parse_args import parse_args, remove_and_create_dir

    args = parse_args()
    args.project_name = 'C3D'
    args.arch = 'cascade_resunet'

    trainer = NetworkTrainer(args)

    save_path = os.path.join(trainer.setting.output_dir, 'onnx')
    onnx_file_path= trainer.setting.onnx_file
    list_patient_dirs = ['../Data/OpenKBP_C3D/pt_' + str(i) for i in range(201, 211)]
    # list_patient_dirs = ['../Data/OpenKBP_C3D/pt_' + str(i) for i in range(241, 341)]
    val_onnx(onnx_file_path, list_patient_dirs, save_path)

    evaluate_OpenKBP(save_path)

    # import onnxruntime
    # providers = onnxruntime.get_available_providers()
    # print(providers)
