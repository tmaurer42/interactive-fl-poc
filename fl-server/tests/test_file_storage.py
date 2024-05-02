import os
import pytest
from storage.file_system_storage import FileSystemStorage

class TestFileStorage:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.storage = FileSystemStorage(folder='__test__file_storage__')
        yield
        for filename in os.listdir(self.storage.folder):
            os.remove(os.path.join(self.storage.folder, filename))
        os.rmdir(self.storage.folder)

    def test_write_and_read(self):
        self.storage.write('test.bin', b'Hello, World!')
        content = self.storage.read('test.bin')
        assert content == b'Hello, World!'

    def test_delete(self):
        self.storage.write('test.bin', b'Hello, World!')
        self.storage.delete('test.bin')
        assert not os.path.exists(os.path.join(self.storage.folder, 'test.bin'))