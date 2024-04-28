import os
from .file_storage_interface import IFileStorage

class FileSystemStorage(IFileStorage):
    def __init__(self, folder: str = '__file_storage__'):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def read(self, file_path: str) -> bytes:
        with open(os.path.join(self.folder, file_path), 'rb') as file:
            return file.read()

    def write(self, file_path: str, content: bytes) -> None:
        with open(os.path.join(self.folder, file_path), 'wb') as file:
            file.write(content)

    def delete(self, file_path: str) -> None:
        os.remove(os.path.join(self.folder, file_path))