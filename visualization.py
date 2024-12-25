import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

path = r'D:\Python_code\DosePrediction\Data\OpenKBP_C3D\pt_126'

paths = os.listdir(path)

ct_image = sitk.ReadImage(os.path.join(path, 'CT.nii.gz'))
dose_image = sitk.ReadImage(os.path.join(path, 'dose.nii.gz'))

ptvs = np.zeros((128, 128, 128), np.uint8)
for i in paths:
    if 'PTV' in i:
        ptv_image = sitk.ReadImage(os.path.join(path, i))
        ptvs += sitk.GetArrayFromImage(ptv_image)

list_oar_names = ['Brainstem',
                  'SpinalCord',
                  'RightParotid',
                  'LeftParotid',
                  'Esophagus',
                  'Larynx',
                  'Mandible']
oars = np.zeros((128, 128, 128), np.uint8)
for i in range(7):
    oar_path = os.path.join(path, list_oar_names[i] + '.nii.gz')
    if os.path.exists(oar_path):
        oar_image = sitk.ReadImage(oar_path)
        oar = sitk.GetArrayFromImage(oar_image)
        oars[oar > 0] = i + 1

ct_array = sitk.GetArrayFromImage(ct_image)
dose_array = sitk.GetArrayFromImage(dose_image)

slice_idx = ct_array.shape[0] // 2
ct_slice = ct_array[slice_idx, :, :]
dose_slice = dose_array[slice_idx, :, :]
ptv_slice = ptvs[slice_idx, :, :]
oar_slice = oars[slice_idx, :, :]

ct_slice_norm = (ct_slice - np.min(ct_slice)) / (np.max(ct_slice) - np.min(ct_slice))
# ct_slice_norm = ct_slice / np.percentile(ct_slice, 99.5)
# ct_slice_norm=np.clip(ct_slice_norm, 0, 1)

dose_slice_norm = dose_slice / np.max(dose_slice)

fig = plt.figure(figsize=(18, 6), dpi=100, tight_layout=True)

fontsize = 18

plt.subplot(1, 3, 1)
plt.imshow(ct_slice_norm, cmap='gray')
plt.contour(ptv_slice, levels=[0.5], colors='red', linewidths=1.5)
plt.contour(oar_slice, levels=[0.5], colors='blue', linewidths=1.5)
plt.title('CT Image Slice', fontsize=fontsize)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(dose_slice_norm, cmap='jet')
plt.title('Dose Image Slice', fontsize=fontsize)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(ct_slice_norm, cmap='gray')
plt.imshow(dose_slice_norm, cmap='jet', alpha=0.3)

plt.title('CT With Dose', fontsize=fontsize)
plt.axis('off')

plt.subplots_adjust(top=0.8)
plt.savefig(f"./visualization.png", dpi=100)

plt.show()

plt.clf()
plt.close(fig)
