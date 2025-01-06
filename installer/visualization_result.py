import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


def visualize_dose(gt_path, pred_path, axis=0, slice_idx=-1):
    ct_image = sitk.ReadImage(os.path.join(gt_path, 'CT.nii.gz'))
    dose_image = sitk.ReadImage(os.path.join(gt_path, 'dose.nii.gz'))
    predict_dose = sitk.ReadImage(pred_path)

    ptvs = np.zeros((128, 128, 128), np.uint8)
    for i in os.listdir(gt_path):
        if 'PTV' in i:
            ptv_image = sitk.ReadImage(os.path.join(gt_path, i))
            ptvs += sitk.GetArrayFromImage(ptv_image)

    oars = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_path, 'OARs.nii.gz')))

    ct_array = sitk.GetArrayFromImage(ct_image)
    dose_array = sitk.GetArrayFromImage(dose_image)
    predict_dose_array = sitk.GetArrayFromImage(predict_dose)

    if slice_idx == -1 or slice_idx > ct_array.shape[axis]:
        slice_idx = ct_array.shape[0] // 2
    print(slice_idx)
    if axis == 0:
        ct_slice = ct_array[slice_idx, :, :]
        dose_slice = dose_array[slice_idx, :, :]
        predict_dose_slice = predict_dose_array[slice_idx, :, :]
        ptv_slice = ptvs[slice_idx, :, :]
        oar_slice = oars[slice_idx, :, :]
    elif axis == 1:
        ct_slice = ct_array[:, slice_idx, :]
        dose_slice = dose_array[:, slice_idx, :]
        predict_dose_slice = predict_dose_array[:, slice_idx, :]
        ptv_slice = ptvs[:, slice_idx, :]
        oar_slice = oars[:, slice_idx, :]
    else:
        ct_slice = ct_array[:, :, slice_idx]
        dose_slice = dose_array[:, :, slice_idx]
        predict_dose_slice = predict_dose_array[:, :, slice_idx]
        ptv_slice = ptvs[:, :, slice_idx]
        oar_slice = oars[:, :, slice_idx]

    ct_slice_norm = (ct_slice - np.min(ct_slice)) / (np.max(ct_slice) - np.min(ct_slice))
    # ct_slice_norm = ct_slice / np.percentile(ct_slice, 99.5)
    # ct_slice_norm=np.clip(ct_slice_norm, 0, 1)

    dose_slice_norm = dose_slice / np.max(dose_slice)
    predict_dose_slice_norm = predict_dose_slice / np.max(predict_dose_slice)

    diff_dose_slice = np.abs(predict_dose_slice - dose_slice)
    # diff_dose_slice = diff_dose_slice / np.max(diff_dose_slice)

    fig = plt.figure(figsize=(10, 6), dpi=100, tight_layout=True)

    fontsize = 15

    plt.subplot(2, 3, 1)
    plt.imshow(ct_slice_norm, cmap='gray')
    plt.contour(ptv_slice, levels=[0.5], colors='red', linewidths=1.5)
    plt.contour(oar_slice, levels=[0.5], colors='blue', linewidths=1.5)
    plt.title('CT Image Slice', fontsize=fontsize)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(dose_slice_norm, cmap='jet')
    plt.title('Dose Image Slice', fontsize=fontsize)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(ct_slice_norm, cmap='gray')
    plt.imshow(dose_slice_norm, cmap='jet', alpha=0.3)
    plt.title('CT with Dose', fontsize=fontsize)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(diff_dose_slice, cmap='jet')
    plt.title('Diff Dose', fontsize=fontsize)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(predict_dose_slice_norm, cmap='jet')
    plt.title('Predicted Dose', fontsize=fontsize)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(ct_slice_norm, cmap='gray')
    plt.imshow(predict_dose_slice_norm, cmap='jet', alpha=0.3)
    plt.title('CT with Predicted Dose', fontsize=fontsize)
    plt.axis('off')

    plt.subplots_adjust(top=0.8)
    # plt.savefig(f"./visualization.png", dpi=300)

    plt.show()

    plt.clf()
    plt.close(fig)


if __name__ == '__main__':
    gt_path = './dist/test_data'
    pred_path = './dist/result/dose.nii.gz'
    # visualize_dose(gt_path, pred_path, 0, 44)
    visualize_dose(gt_path, pred_path, -1)
