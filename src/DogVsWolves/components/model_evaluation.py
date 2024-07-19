import torch
from DogVsWolves.components.model_manager import ImageDataset


class EvaluateModel:
    def __init__(self, config, model) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def eval(self, data):
        pass

    
