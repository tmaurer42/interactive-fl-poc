import os
from typing import Optional

import torch
import numpy as np
import onnx
from onnxruntime.training import artifacts, onnxblock


def generate_training_artifacts(
    trainable_param_names: list[str],
    model_file_path: bytes,
):
    onnx_model = onnx.load_model(model_file_path)
    model_directory = os.path.dirname(model_file_path)

    artifacts.generate_artifacts(
        onnx_model,
        requires_grad=trainable_param_names,
        loss=artifacts.LossType.CrossEntropyLoss,
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=model_directory,
    )

    # We need to delete the two last outputs (running_mean, running_var)
    # from the BatchNorm layers in the eval model, keeping only the first output.
    # Else, loading the training session will throw an error.
    eval_model_path = os.path.join(model_directory, "eval_model.onnx")
    eval_model = onnx.load(eval_model_path)
    for node in eval_model.graph.node:
        if node.op_type == "BatchNormalization":
            del node.output[1:]
    onnx.save(eval_model, eval_model_path)


def model_to_onnx(
    model: torch.nn.Module,
    model_directory: str,
    base_model_name: Optional[str] = "model",
    training_artifacts: Optional[bool] = True
):
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    base_model_file_name = f"{base_model_name}.onnx"
    file_path = os.path.join(model_directory, base_model_file_name)
    x = torch.randn(1, 3, 224, 224)

    model.eval()

    torch.onnx.export(
        model,
        x,
        f=file_path,
        export_params=True,
        training=torch.onnx.TrainingMode.TRAINING,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},
        opset_version=14,
    )

    trainable_param_names = [
        name for name, param in model.named_parameters() if param.requires_grad]

    if training_artifacts:
        generate_training_artifacts(
            trainable_param_names,
            file_path
        )


def get_parameters(
    model_file: bytes,
    trainable_param_names: list[str]
) -> list[float]:
    model = onnx.load_from_string(model_file)
    layers = model.graph.initializer

    parameters = []
    for layer in layers:
        if not layer.name in trainable_param_names:
            continue

        tensor_array = onnx.numpy_helper.to_array(layer).flatten().tolist()
        parameters.extend(tensor_array)
    
    return parameters


def set_parameters(
    model_file: bytes,
    params: list[float],
    trainable_param_names: list[str]
):
    model = onnx.load_from_string(model_file)
    layers = model.graph.initializer

    param_index = 0
    for layer in layers:
        if not layer.name in trainable_param_names:
            continue

        tensor_array = onnx.numpy_helper.to_array(layer)
        num_elements = tensor_array.size
        new_values = np.array(
            params[param_index:param_index + num_elements], dtype=tensor_array.dtype)
        new_values = new_values.reshape(tensor_array.shape)

        updated_layer = onnx.numpy_helper.from_array(new_values, layer.name)
        layer.CopyFrom(updated_layer)

        param_index += num_elements

    return model
