import numpy as np
import onnx
import onnxruntime
import torch

from network_trainer import NetworkTrainer


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def convert_onnx(trainer):
    trainer.setting.network.eval()
    x = torch.rand(1, 3, trainer.setting.volume_size[0], trainer.setting.volume_size[1], trainer.setting.volume_size[2],
                   requires_grad=True).to(trainer.setting.device)

    # for module in trainer.setting.network.modules():
    #     if isinstance(module, torch.nn.InstanceNorm3d):
    #         print(module)
    #         module.train(False)

    torch_out = trainer.setting.network(x)

    torch.onnx.export(trainer.setting.network,
                      x,
                      trainer.setting.onnx_file,
                      input_names=['input'],
                      output_names=['output'],
                      verbose=False)
    try:
        onnx_model = onnx.load(trainer.setting.onnx_file)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model is valid!")
    except Exception as e:
        print(f"Error checking ONNX model: {e}")
        return

    ort_session = onnxruntime.InferenceSession(trainer.setting.onnx_file, providers=["CPUExecutionProvider"])
    # for input in ort_session.get_inputs():
    #     print(f"Input name: {input.name}, shape: {input.shape}")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

    ort_outs = ort_session.run(None, ort_inputs)
    for i in range(len(ort_outs)):
        print(f"Output {i}: {ort_outs[i].shape}")
    # print(to_numpy(torch_out[-1])- ort_outs[-1])
    np.testing.assert_allclose(to_numpy(torch_out[-1]), ort_outs[-1], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()
    args.project_name = 'C3D'
    args.arch = 'cascade_resunet'

    trainer = NetworkTrainer(args)
    trainer.init_trainer(ckpt_file=trainer.setting.best_ckpt_file, only_network=True)
    convert_onnx(trainer)
