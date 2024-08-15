from typing import Literal
from torch import nn
from torchvision import models, transforms


ModelType = Literal["MobileNetV2"]


def get_model_for_training(
    model_type: ModelType, 
    num_classes: int, 
    transfer_learning: bool, 
    dropout: float
):
    if model_type == 'MobileNetV2':
        return get_mobilenet_for_training(num_classes, transfer_learning, dropout)


def get_model_for_inference(
    model_type: ModelType
):
    if model_type == 'MobileNetV2':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        return models.mobilenet_v2(weights=weights)


def get_mobilenet_for_training(
    num_classes: int,
    transfer_learning: bool = False,
    dropout=0.2,
) -> models.MobileNetV2:
    """
    Load and initialize a pytorch MobileNetV2 model.

    Parameters:
        num_classes (int):
            Number of classes the model should output in the last layer.
        transfer_learning (bool):
            If True, initialize the model for transfer learning. 
            First, freeze all base layers and then 
            add two fully connected layers with dropout before 
            the classification layer. 
            Default is False.
        dropout (float): 
            Value for the dropout layers. 
            This affects the default dropout layer and the ones added when transfer_learning is True.
            Default is 0.2.

    Returns:
        The configured MobileNetV2 model.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if transfer_learning else None
    mobilenetv2_model = models.mobilenet_v2(weights=weights)

    last_layer_input_size = mobilenetv2_model.last_channel
    layers = []

    if transfer_learning:
        for params in mobilenetv2_model.parameters():
            params.requires_grad = False

        dense_layer_size = last_layer_input_size // 2
        dense_layer_2_size = last_layer_input_size // 4
        layers.extend([
            nn.Linear(last_layer_input_size, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_layer_size, dense_layer_2_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        last_layer_input_size = dense_layer_2_size

    layers.append(nn.Linear(last_layer_input_size, num_classes))

    mobilenetv2_model.classifier[0] = nn.Dropout(dropout)
    mobilenetv2_model.classifier[1] = nn.Sequential(*layers)

    return mobilenetv2_model