from abc import ABC
from dataclasses import dataclass, field
from typing import Literal

import onnx

from ml import onnx_utils
from ml.onnx_utils import get_parameters, set_parameters
from storage.file_storage_interface import IFileStorage


@dataclass
class FLTaskBase(ABC):
    """
    Holds all information and instructions for a Federated Learning task.
    """
    id: str
    title: str
    aggregator: Literal["fedasync"]
    aggregator_params: dict[str, any]

    model_file: str
    training_file: str
    optimizer_file: str
    eval_file: str
    checkpoint_file: str

    batch_size: int
    local_epochs: int

    model_version: int
    trainable_parameter_names: list[str] = field(
        init=False, default_factory=list)

    def aggregate_fedasync(
        self,
        params: list[float],
        update: list[float],
        version: int
    ) -> list[float]:
        mixing_param: float = self.aggregator_params['mixing_param']
        staleness = self.model_version - version
        alpha = mixing_param / (staleness + 1.0)

        new_params = [(1-alpha)*w + alpha*w_update for w,
                      w_update in zip(params, update)]

        return new_params

    def handleUpdate(
        self,
        update: list[float],
        version: int,
        storage: IFileStorage
    ):
        """
        Handles an incoming **update** from a client, trained with the 
        specified model **version** and applies it to the global model, 
        using the tasks 'aggregator' algorithm.
        Then, the model with the new parameters is saved to the storage and
        new training artifacts are created.
        """
        model_bytes = storage.read(self.model_file)
        model_file_path = storage.get_full_path(self.model_file)

        params = get_parameters(model_bytes, self.trainable_parameter_names)

        if self.aggregator == "fedasync":
            new_params = self.aggregate_fedasync(params, update, version)
        else:
            raise NotImplementedError(
                f"Aggregator '${self.aggregator}' is not implemented")

        model = set_parameters(
            model_bytes, new_params, self.trainable_parameter_names)

        onnx.save_model(model, model_file_path)
        onnx_utils.generate_training_artifacts(
            self.trainable_parameter_names, model_file_path)

        self.model_version += 1


@dataclass
class ClassificationFLTask(FLTaskBase):
    classes: list[str]
    input_size: int
    norm_range: list[int]
