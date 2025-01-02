import os
import shutil
import sys

import SimpleITK as sitk
import numpy as np
import onnxruntime
from tqdm import tqdm

from evaluate_openKBP import evaluate_OpenKBP


def save_dose(array, file_path, reference=None):
    image = sitk.GetImageFromArray(array)
    image = sitk.Cast(image, sitk.sitkFloat32)
    if reference is not None:
        image.CopyInformation(reference)
    sitk.WriteImage(image, file_path)


def read_data(patient_dir):
    if not os.path.exists(patient_dir):
        raise ValueError('Patient directory does not exist!')
    dict_images = {}
    list_structures = ['CT',
                       'PTV70',
                       'PTV63',
                       'PTV56',
                       'possible_dose_mask',
                       'OARs']

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
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)
    return dict_images


def img_normalize(img):
    min_value = np.min(img)
    max_value = np.max(img)
    img = (img - min_value) / (max_value - min_value + 1e-8)
    return img


def pre_processing(dict_images):
    PTVs = 70.0 / 70. * dict_images['PTV70'] \
           + 63.0 / 70. * dict_images['PTV63'] \
           + 56.0 / 70. * dict_images['PTV56']
    OAR_all = dict_images['OARs']

    CT = dict_images['CT'].astype(np.float32)
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = img_normalize(CT)


    possible_dose_mask = dict_images['possible_dose_mask']
    list_images = [np.concatenate((PTVs, OAR_all, CT), axis=0),
                   possible_dose_mask]
    return list_images


def val_onnx(onnx_file_path, list_patient_dirs, save_path):
    remove_and_create_dir(save_path)
    session = onnxruntime.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])

    for patient_dir in tqdm(list_patient_dirs, file=sys.stdout):
        patient_id = patient_dir.split('/')[-1]

        dict_images = read_data(patient_dir)
        list_images = pre_processing(dict_images)

        input = list_images[0]
        possible_dose_mask = list_images[-1]

        input = np.expand_dims(input, 0).astype(np.float32)
        ort_inputs = {session.get_inputs()[0].name: (input)}
        result = session.run(None, ort_inputs)
        prediction = np.squeeze(result[-1])
        # prediction[prediction < 0] = 0

        prediction[np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)] = 0
        prediction = 70. * prediction

        template_nii = sitk.ReadImage(patient_dir + '/CT.nii.gz')
        result_path = os.path.join(save_path, patient_id)
        os.makedirs(result_path, exist_ok=True)
        dose_path = os.path.join(save_path, patient_id, 'dose.nii.gz')
        # print(f'Save in {result_path}...')
        save_dose(prediction, dose_path, template_nii)


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    save_path = os.path.join('./', 'onnx')
    onnx_file_path= r'D:\Python_code\DosePrediction\Output\C3D\dose_prediction.onnx'
    list_patient_dirs = ['../Data/OpenKBP_C3D/pt_' + str(i) for i in range(201, 202)]
    # list_patient_dirs = ['../Data/OpenKBP_C3D/pt_' + str(i) for i in range(241, 341)]
    val_onnx(onnx_file_path, list_patient_dirs, save_path)
    evaluate_OpenKBP(save_path)
