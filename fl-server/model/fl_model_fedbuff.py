from dataclasses import dataclass
from .file import File
from .fl_model_base import FLModel

@dataclass
class FedBuffFLModel(FLModel):
    def handleUpdate(self, update):
        pass