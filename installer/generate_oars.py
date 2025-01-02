import os
import SimpleITK as sitk
import numpy as np


def read_oars(patient_dir):
    if not os.path.exists(patient_dir):
        raise ValueError(f'Patient directory "{patient_dir}" does not exist!')
    else:
        print(f'Processing patient directory: {patient_dir}')

    structures = [
        'Brainstem', 'SpinalCord', 'RightParotid',
        'LeftParotid', 'Esophagus', 'Larynx', 'Mandible'
    ]

    masks = []

    for structure_name in structures:
        structure_file = os.path.join(patient_dir, f'{structure_name}.nii.gz')
        if os.path.exists(structure_file):
            mask = sitk.ReadImage(structure_file, sitk.sitkUInt8)
            mask_array = sitk.GetArrayFromImage(mask)[np.newaxis, :, :, :]
        else:
            mask_array = np.zeros((1, 128, 128, 128), dtype=np.uint8)

        masks.append(mask_array)

    combined_mask = np.any(masks, axis=0).astype(np.uint8)

    save_array_as_nii(combined_mask[0], os.path.join(patient_dir, 'OARs.nii.gz'), mask)


def process_patients(patient_dirs):
    for patient_dir in patient_dirs:
        try:
            read_oars(patient_dir)
        except ValueError as e:
            print(e)

def save_array_as_nii(array, file_path, reference=None):
    image = sitk.GetImageFromArray(array)
    image = sitk.Cast(image, sitk.sitkUInt8)
    if reference is not None:
        image.CopyInformation(reference)
    sitk.WriteImage(image, file_path)


if __name__ == '__main__':
    list_patient_dirs = [f'../Data/OpenKBP_C3D/pt_{i}' for i in range(1, 341)]

    process_patients(list_patient_dirs)
