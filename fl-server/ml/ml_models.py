from typing import Literal

from ml.mobilenet import get_mobilenet

MLModel = Literal["MobileNetV2"]

def get_ml_model(
    kind: MLModel, 
    num_classes: int,
):
    if kind == "MobileNetV2":
        return get_mobilenet(num_classes)