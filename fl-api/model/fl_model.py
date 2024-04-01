class FLModel:
    def __init__(self, 
                 model,
                 algorithm='fedavg',):
        self.model = model
        self.algorithm = algorithm