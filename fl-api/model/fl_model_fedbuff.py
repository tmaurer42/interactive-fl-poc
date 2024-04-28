from .file import File
from .fl_model_base import FLModel

class FedBuffFLModel(FLModel):
    def __init__(self, 
                 title: str,
                 file: File,):
        super().__init__(title, file)

        self.buffer = None

    def receiveUpdate(self, update):
        pass