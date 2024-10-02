import os
from .file_storage_interface import IFileStorage

class FileSystemStorage(IFileStorage):
    def __init__(self, folder: str = '__file_storage__'):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def get_full_path(self, file_path) -> str:
        return os.path.join(self.folder, file_path)

    def read(self, file_path: str) -> bytes:
        with open(os.path.join(self.folder, file_path), 'rb') as file:
            return file.read()

    def write(self, file_path: str, content: bytes) -> None:
        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(os.path.join(self.folder, file_path), 'wb') as file:
            file.write(content)

    def delete(self, file_path: str) -> None:
        os.remove(os.path.join(self.folder, file_path))