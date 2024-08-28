from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class FLModel(ABC):
    id: str
    title: str
    classes: list[str]
    file: str
    training_file: str
    optimizer_file: str
    eval_file: str
    checkpoint_file: str
    input_size: int
    norm_range: list[int]

    version: int = 0

    @abstractmethod
    def handleUpdate(self, update):
        pass