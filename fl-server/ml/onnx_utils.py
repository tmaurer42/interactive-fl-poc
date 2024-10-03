import os
from typing import Optional

import torch
import numpy as np
import onnx
from onnxruntime.training import artifacts


def generate_training_artifacts(
    trainable_param_names: list[str],
    model_file_path: str
):
    """
    Generates training artifacts based on the model at the given file path.
    Trainable parameter names need to be provided explicitly.
    """
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
    # Else, loading the training session on the client will throw an error.
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
    """
    Exports a pytorch model to onnx in evaluation model.
    Optionally, training articatcs for on-device training 
    can be created in the same storage location.
    """
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
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}},
        opset_version=14,
    )

    if training_artifacts:
        train_base_model_file = os.path.join(model_directory, f"train_base_{base_model_file_name}")
        torch.onnx.export(
            model,
            x,
            f=train_base_model_file,
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

        generate_training_artifacts(
            trainable_param_names,
            train_base_model_file
        )


def get_parameters(
    model_file: bytes,
    trainable_param_names: list[str]
) -> list[float]:
    """
    Returns the parameters as a 1d array from an onnx model (as bytes).
    Only the parameters from the provided trainable param names are returned.
    """
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
    """
    Sets parameters in a 1d array in the given onnx model (in bytes) in
    the given trainable param names.
    Returns the model with the new parameters.
    """
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
