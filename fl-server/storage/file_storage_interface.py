from abc import ABC, abstractmethod

class IFileStorage(ABC):
    @abstractmethod
    def get_full_path(self, file_path) -> str:
        """Gets the full path directly to the file"""
        pass

    @abstractmethod
    def read(self, file_path: str) -> str:
        """Reads the file at the given path and returns its contents."""
        pass

    @abstractmethod
    def write(self, file_path: str, content: str) -> None:
        """Writes the given content to the file at the given path."""
        pass

    @abstractmethod
    def delete(self, file_path: str) -> None:
        """Deletes the file at the given path."""
        pass