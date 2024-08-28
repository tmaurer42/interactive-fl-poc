from abc import ABC, abstractmethod
from dataclasses import dataclass
from .file import File

@dataclass
class FLModel(ABC):
    id: str
    title: str
    file: File
    training_file: File
    optimizer_file: File
    eval_file: File
    checkpoint_file: File
    input_size: int
    norm_range: list[int]

    version: int = 0

    @abstractmethod
    def handleUpdate(self, update):
        pass