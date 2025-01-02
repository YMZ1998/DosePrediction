import argparse
import os
import shutil
import time

import SimpleITK as sitk
import numpy as np
import onnxruntime


def save_dose(array, file_path, reference=None):
    image = sitk.GetImageFromArray(array)
    image = sitk.Cast(image, sitk.sitkFloat32)
    if reference is not None:
        image.CopyInformation(reference)
    sitk.WriteImage(image, file_path)


def read_data(args):
    dict_images = {}
    list_structures = {'CT': args.ct_path,
                       'PTV70': args.ptvs70_path,
                       'PTV63': args.ptvs63_path,
                       'PTV56': args.ptvs56_path,
                       'possible_dose_mask': args.possible_dose_mask_path,
                       'OARs': args.oars_path, }

    for structure_name in list_structures.keys():
        structure_file = list_structures[structure_name]
        if structure_name == 'CT':
            dtype = sitk.sitkInt16
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
    PTVs = 70.0 / 70. * dict_images['PTV70'] + 63.0 / 70. * dict_images['PTV63'] + 56.0 / 70. * dict_images['PTV56']
    OAR_all = dict_images['OARs']

    CT = dict_images['CT'].astype(np.float32)
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = img_normalize(CT)

    possible_dose_mask = dict_images['possible_dose_mask']
    list_images = [np.concatenate((PTVs, OAR_all, CT), axis=0), possible_dose_mask]
    return list_images


def val_onnx(args):
    start_time = time.time()
    session = onnxruntime.InferenceSession(args.onnx_file_path, providers=["CPUExecutionProvider"])

    dict_images = read_data(args)
    list_images = pre_processing(dict_images)

    input = list_images[0]
    possible_dose_mask = list_images[-1]

    input = np.expand_dims(input, 0).astype(np.float32)
    ort_inputs = {session.get_inputs()[0].name: (input)}
    print("Infer...")
    result = session.run(None, ort_inputs)
    prediction = np.squeeze(result[-1])

    prediction[np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)] = 0
    prediction = 70. * prediction

    template_nii = sitk.ReadImage(args.ct_path)
    os.makedirs(args.result_path, exist_ok=True)
    dose_path = os.path.join(args.result_path, 'dose.nii.gz')
    print(f'Save in {dose_path}')
    save_dose(prediction, dose_path, template_nii)
    total_time = time.time() - start_time
    print("time {}s".format(total_time))

def parse_arguments():
    parser = argparse.ArgumentParser(prog='DosePrediction.py',
                                     description="Dose prediction script for radiotherapy planning.")
    parser.add_argument('--onnx_file_path', type=str, default='./checkpoint/dose_prediction.onnx',
                        help="Path to the ONNX model file (default: './checkpoint/dose_prediction.onnx')")
    parser.add_argument('--ct_path', type=str, required=True, help="Path to the CT file.")
    parser.add_argument('--ptvs70_path', type=str, required=True, help="Path to the PTV70 mask file.")
    parser.add_argument('--ptvs63_path', type=str, required=True, help="Path to the PTV63 mask file.")
    parser.add_argument('--ptvs56_path', type=str, required=True, help="Path to the PTV56 mask file.")
    parser.add_argument('--oars_path', type=str, required=True, help="Path to the OARs mask file.")
    parser.add_argument('--possible_dose_mask_path', type=str, required=True,
                        help="Path to the possible dose mask file.")
    parser.add_argument('--result_path', type=str, default='./result',
                        help="Path to save prediction results (default: './result').")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    val_onnx(args)
