from abc import ABC, abstractmethod
from dataclasses import dataclass
from .file import File

@dataclass
class FLModel(ABC):
    title: str
    file: File
    input_size: int
    norm_range: tuple[int, int]

    @abstractmethod
    def handleUpdate(self, update):
        pass