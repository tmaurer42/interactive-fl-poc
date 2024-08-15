import argparse
import os

import torch
from torch import nn

from models import ModelType, get_model_for_training, get_model_for_inference


MODELS_FOLDER = 'onnx_models'


def main(
    name: str, 
    model_type: ModelType, 
    num_classes: int, 
    transfer_learning: bool, 
    dropout: float, 
    imagenet_inference_only: bool
):
    if imagenet_inference_only:
        model = get_model_for_inference(model_type)
    else:
        model = get_model_for_training(
            model_type, num_classes, transfer_learning, dropout
        )

    batch_size = 64     # just a random number
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    base_path = MODELS_FOLDER
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    model_file = os.path.join(base_path, f"{name}.onnx")

    torch.onnx.export(model,        # model being run
        x,                          # model input (or a tuple for multiple inputs)
        model_file,                 # where to save the model (can be a file or file-like object)
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=20,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names = ['input'],    # the model's input names
        output_names = ['output'],  # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3: 'width'},    # variable length axes
                    'output' : {0 : 'batch_size'}})

    print(model_type, num_classes, transfer_learning, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get an ONNX model")

    parser.add_argument('--name', type=str, required=True,
                        help=f"Name of the model for the output file")
    model_type_choices: list[ModelType] = ['MobileNetV2']
    parser.add_argument('--model_type', type=str, required=True, choices=model_type_choices,
                        help=f"Type of model to use")
    parser.add_argument('--imagenet_inference_only', action='store_true', required=False, default=True,
                        help=f"Type of model to use")
    parser.add_argument('--num_classes', type=int, required=False, default=2,
                        help=f"Number of classes in the classifier")
    parser.add_argument('--transfer_learning', action='store_true', required=False, default=False,
                        help='Whether to use transfer learning or not.')
    parser.add_argument('--dropout', type=float, required=False, default=0.2,
                        help=f"The dropout rate of the model")

    args = parser.parse_args()

    main(args.name, args.model_type, args.num_classes, args.transfer_learning, args.dropout, args.imagenet_inference_only)