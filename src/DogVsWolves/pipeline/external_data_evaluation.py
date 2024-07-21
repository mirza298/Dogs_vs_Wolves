import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from DogVsWolves.components.model_manager import ImageDataset, ConvolutionalNeuralNetwork, plot_confusion_matrix

class ExternalDataEvaluation:
    def __init__(self, config) -> None:
        self.config = config
        self.transform = v2.Compose([
            v2.Resize(
                size=(self.config.params_image_size, 
                      self.config.params_image_size), antialias=True),
                      v2.ToDtype(torch.float32, scale=True),
                      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Pytorch standard
                      ])


    def evaluate(self):
        data = ImageDataset(self.config.data_dir, self.transform)
        data_loader = DataLoader(data)

        loss_function = torch.nn.CrossEntropyLoss()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ConvolutionalNeuralNetwork()
        model.load_state_dict(torch.load(self.config.trained_model_inference_path))
        model = model.to(device)

        loss, accuracy, y_pred, y_true = model.evaluate_model(data_loader, device, loss_function)

        print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.4f} ")

        plot_confusion_matrix(y_true, y_pred, data.image_labels_mapping)