# id = '../Data/OpenKBP_C3D/pt_201'
id = r'./test_data'

list_structures = ['CT',
                   'PTV70',
                   'PTV63',
                   'PTV56',
                   'possible_dose_mask',
                   'OARs']
paths = {}
for structure_name in list_structures:
    structure_file = id + '/' + structure_name + '.nii.gz'
    paths[structure_name] = structure_file

print(f"python DosePrediction.py --ct_path {paths['CT']} --ptvs70_path {paths['PTV70']} --ptvs63_path {paths['PTV63']} --ptvs56_path {paths['PTV56']} --oars_path {paths['OARs']} --possible_dose_mask_path {paths['possible_dose_mask']}")

print(f"DosePrediction.exe --ct_path {paths['CT']} --ptvs70_path {paths['PTV70']} --ptvs63_path {paths['PTV63']} --ptvs56_path {paths['PTV56']} --oars_path {paths['OARs']} --possible_dose_mask_path {paths['possible_dose_mask']}")
