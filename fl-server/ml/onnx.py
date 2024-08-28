import os
from typing import Optional
import torch
import onnx
from onnxruntime.training import artifacts


def generate_training_artifacts(
    model: torch.nn.Module,
    model_directory: str, 
    base_model_file_name: str
):
    onnx_model = onnx.load_model(os.path.join(model_directory, base_model_file_name))
    requires_grad = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]


    # Generate the training artifacts
    artifacts.generate_artifacts(
        onnx_model,
        requires_grad=requires_grad,
        frozen_params=frozen_params,
        loss=artifacts.LossType.CrossEntropyLoss,
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=model_directory
    )


def model_to_onnx(
    model: torch.nn.Module, 
    model_directory: str,
    base_model_name: Optional[str] = "model",
    training_artifacts: Optional[bool] = True
): 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    base_model_file_name = f"{base_model_name}.onnx"

    file = os.path.join(model_directory, base_model_file_name)

    batch_size = 1     # just a random number
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=False)

    torch.onnx.export(
        model,
        x,
        f=file,
        export_params=True,
        training=torch.onnx.TrainingMode.TRAINING,
        opset_version=20,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )

    if training_artifacts:
        generate_training_artifacts(
            model,
            model_directory,
            base_model_file_name
        )