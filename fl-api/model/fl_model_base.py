from abc import ABC, abstractmethod
from .file import File

class FLModel(ABC):
    def __init__(self, 
                 title: str,
                 file: File,):
        self.title: str = title
        self.file: File = file

    @abstractmethod
    def receiveUpdate(self, update):
        pass