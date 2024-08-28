from dataclasses import dataclass
from .fl_model_base import FLModel

@dataclass
class FedBuffFLModel(FLModel):
    def handleUpdate(self, update):
        pass