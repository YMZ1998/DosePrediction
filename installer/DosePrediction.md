# DosePrediction

---

## DosePrediction.exe 命令行参数说明

### 1. `--onnx_file_path`
- **类型**: `str`
- **默认值**: `'./checkpoint/dose_prediction.onnx'`
- **描述**: ONNX 模型文件的路径。该文件用于加载预训练的辐射治疗剂量预测模型。如果没有指定该参数，将使用默认路径 `'./checkpoint/dose_prediction.onnx'`。

### 2. `--ct_path`
- **类型**: `str`
- **必需**: 是
- **描述**: CT 图像文件的路径。该路径应指向一张经过处理的计算机断层扫描（CT）图像，用于辐射治疗计划中的剂量预测。

### 3. `--ptvs70_path`
- **类型**: `str`
- **必需**: 是
- **描述**: PTV70 掩膜文件的路径。该路径指向 PTV70（计划靶区 70）掩膜文件，用于表示计划靶区中目标剂量的区域。

### 4. `--ptvs63_path`
- **类型**: `str`
- **必需**: 是
- **描述**: PTV63 掩膜文件的路径。该路径指向 PTV63（计划靶区 63）掩膜文件，用于表示计划靶区中目标剂量的区域。

### 5. `--ptvs56_path`
- **类型**: `str`
- **必需**: 是
- **描述**: PTV56 掩膜文件的路径。该路径指向 PTV56（计划靶区 56）掩膜文件，用于表示计划靶区中目标剂量的区域。

### 6. `--oars_path`
- **类型**: `str`
- **必需**: 是
- **描述**: OARs 掩膜文件的路径。该路径指向 OARs（关键器官的保护区）掩膜文件，用于表示在辐射治疗过程中需要保护的器官区域。

### 7. `--possible_dose_mask_path`
- **类型**: `str`
- **必需**: 是
- **描述**: 可能剂量掩膜文件的路径。该路径指向可能的剂量分布掩膜文件，用于标记在治疗中可能接受辐射剂量的区域。

### 8. `--result_path`
- **类型**: `str`
- **默认值**: `'./result'`
- **描述**: 预测结果保存路径。该路径用于保存预测的结果。如果未指定该参数，将使用默认路径 `'./result'`。

---

### 示例命令

假设你要运行 `DosePrediction.exe`，以下是一个命令行示例：

```bash
DosePrediction.exe \
  --ct_path ./test_data/CT.nii.gz \
  --ptvs70_path ./test_data/PTV70.nii.gz \
  --ptvs63_path ./test_data/PTV63.nii.gz \
  --ptvs56_path ./test_data/PTV56.nii.gz \
  --oars_path ./test_data/OARs.nii.gz \
  --possible_dose_mask_path ./test_data/possible_dose_mask.nii.gz

```

---

### 说明
- **必需参数**: 其中的 `--ct_path`、`--ptvs70_path`、`--ptvs63_path`、`--ptvs56_path`、`--oars_path` 和 `--possible_dose_mask_path` 是必需的参数，必须在运行脚本时提供。
- **默认值**: 如果没有提供 `--onnx_file_path` 和 `--result_path` 参数，脚本会使用默认路径 `./checkpoint/dose_prediction.onnx` 和 `./result`。
