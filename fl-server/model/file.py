import mimetypes

class File:
    def __init__(self, path):
        self.path = path
        self.mimetype = mimetypes.guess_type(path)[0] or 'application/octet-stream'