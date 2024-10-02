from torch import nn
from torchvision import models

def get_mobilenet(
    num_classes: int,
    transfer_learning: bool = False,
    dropout=0.2,
) -> models.MobileNetV2:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if transfer_learning else None
    mobilenetv2_model = models.mobilenet_v2(weights=weights)

    last_layer_input_size = mobilenetv2_model.last_channel
    layers = []

    if transfer_learning:
        for params in mobilenetv2_model.parameters():
            params.requires_grad = False

    layers.append(nn.Linear(last_layer_input_size, num_classes))

    mobilenetv2_model.classifier[0] = nn.Dropout(dropout)
    mobilenetv2_model.classifier[1] = nn.Sequential(*layers)

    return mobilenetv2_model