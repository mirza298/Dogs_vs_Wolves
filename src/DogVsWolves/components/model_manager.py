import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.io import read_image
from torchvision.transforms import v2
import torch.nn as nn
from DogVsWolves.entity.config_entity import ModelConfig
from pathlib import Path
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_directory, transform = None):
        self.image_directory = image_directory
        self.transform = transform
        self.image_labels = os.listdir(image_directory)
        self.image_labels_mapping = {k:v for k, v in zip(self.image_labels, range(len(self.image_labels)))}
        
        # Loop through image_directory and store tuple (image-file-name, image-label) in a list:
        self.image_label_data_pairs = list()
        for path, subdirs, files in os.walk(image_directory):
            for name in files:
                image_path = os.path.join(path, name)
                image_label = self.image_labels_mapping[os.path.basename(path)]
                self.image_label_data_pairs.append((image_path, image_label))
        
    def __len__(self):
        return len(self.image_label_data_pairs)
    
    def __getitem__(self, idx):
        path, label = self.image_label_data_pairs[idx]
        image = read_image(path)
        if self.transform:
            image = self.transform(image)
        #label = torch.tensor(label, dtype = torch.long)
        return image, label
    

class PrepareData:
    def __init__(self, config: ModelConfig):
         self._config = config
    
    def split(self):
        if self._config.params_is_augmentation:
            self._transform = v2.Compose([
                v2.Resize(
                    size=(self._config.params_image_size, 
                          self._config.params_image_size), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ColorJitter(),
                v2.RandomErasing(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Pytorch standard
            ])
        else:
            self._transform = None

        # Create Dataset instance
        data = ImageDataset(image_directory=self._config.data, transform=self._transform)
    
        # Define DataLoader for train(70%), validation(20%) and test(10%) dataset
        self._dataset_size = len(data)
        indices = list(range(self._dataset_size))
        train_split = int(np.floor(self._config.params_train_size * self._dataset_size))
        validation_split = int(np.floor(self._config.params_validation_size * self._dataset_size))

        np.random.shuffle(indices)

        self._train_indices = indices[:train_split]
        self._validation_indices = indices[train_split:(train_split + validation_split)]
        self._test_indices = indices[(train_split + validation_split):]

        # Define DataLoaders for each dataset
        self._train_sampler = SubsetRandomSampler(self._train_indices)
        self._valid_sampler = SubsetRandomSampler(self._validation_indices)
        self._test_sampler = SubsetRandomSampler(self._test_indices)

        self.train_loader = DataLoader(data, batch_size = self.config.params_batch_size, sampler = self._train_sampler)
        self.validation_loader = DataLoader(data, batch_size = self.config.params_batch_size, sampler = self._valid_sampler)
        self.test_loader = DataLoader(data, batch_size = 1, sampler = self._test_sampler)

        return self.train_loader, self.validation_loader, self.test_loader

    

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.config = config

        # First Convolution Layer
        # Input size: [128, 128, 3]
        # Output size: [1+(128 + 2 * padding - kernel_size)/stride,
        #              1+(128 + 2 * padding - kernel_size)/stride,
        #              out_channels]
        # padding: 0, stride: 1, kernel_size: 5
        # Input: 128 x 128 x 3
        # Output: 62 x 62 x 4
        self._first_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=self.config.params_channels*1,
                      kernel_size=self.config.params_kernel_size_cl,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.config.params_channels*1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
        )
        
        # Second Convolution Layer
        # Input: 62 x 62 x 4
        # Output: 30 x 30 x 8
        self._second_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=self.config.params_channels*1,
                      out_channels=self.config.params_channels*2,
                      kernel_size=self.config.params_kernel_size_cl,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.config.params_channels*2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
        )
        
        
        # Third Convolution Layer
        # Input: 30 x 30 x 8
        # Output: 14 x 14 x 12
        self._third_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=self.config.params_channels*2,
                      out_channels=self.config.params_channels*3,
                      kernel_size=self.config.params_kernel_size_cl,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.config.params_channels*3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
        )
        
        # Fourth Convolution Layer
        # Input: 14 x 14 x 12
        # Output: 5 x 5 x 16
        self._fourth_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels=self.config.params_channels*3,
                      out_channels=self.config.params_channels*4,
                      kernel_size=self.config.params_kernel_size_cl,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.config.params_channels*4),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
        )
        
        # Linear (FC) Layer - Otuput
        # Input: 5 x 5 x 16
        # Output: 2
        self._fc_block = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features = 2*4*16),
            nn.ReLU(),
            nn.Linear(in_features=2*4*16, out_features = 2)
        )

        
    def forward(self, x):
        # First Block forward pass
        out = self._first_convolution_block(x)
        
        # Second Block forward pass
        out = self._second_convolution_block(out)
        
        # Third Block forward pass
        out = self._third_convolution_block(out)
        
        # Fourth Block forward pass
        out = self._fourth_convolution_block(out)
        
        # Output
        out = self._fc_block(out)
        
        return out
    

class ModelTrainer:
    def __init__(self, config: ModelConfig, model: ConvolutionalNeuralNetwork):
        self.config = config
        self.model = model
    
    def _train_step(self):
        # 1.) Set model mode to train
        self.model.train()
        
        # 2.) Create variables for loss and accuracy
        loss_results, acc_results = 0, 0
        
        # 3.) Loop throught batches
        for i, (X, y) in enumerate(self.train_loader):
            # 3.1) Move data to device, model has to be on same device!
            X, y = X.to(self.device), y.to(self.device)
            
            # 3.2) Forward pass
            y_predictions = self.model(X)
            
            # 3.3) Calculate loss and acc
            loss = self.loss_function(y_predictions, y)
            loss_results += loss.data.item()
            y_predictions_label = torch.argmax(torch.softmax(y_predictions, dim = 1), dim = 1)
            acc_results += (y_predictions_label == y).sum().item() / len(y)
            
            # 3.4) Empty gradients
            self.optimizer.zero_grad()
            
            # 3.5) Backward pass
            loss.backward()
            
            # 3.6) Update weights and biases
            self.optimizer.step()
        
        # 4.) Retrun loss and accuracy
        return loss_results / len(self.train_loader), acc_results / len(self.train_loader)
    
    def _evaluation_step(self, data_loader: DataLoader):
        # 1.) Set model mode to evaluate mode
        self.model.eval()
        
        # 2.) Create variables for loss and accuracy
        loss_results, acc_results = 0, 0
        
        with torch.no_grad():
            # 3.) Loop throught batches
            for i, (X, y) in enumerate(data_loader):
                # 3.1) Move data to device, model has to be on same device!
                X, y = X.to(self.device), y.to(self.device)

                # 3.2) Forward pass
                y_predictions = self.model(X)
                
                # 3.3) Calculate loss and acc
                loss = self.loss_function(y_predictions, y)
                loss_results += loss.data.item()
                y_predictions_label = torch.argmax(torch.softmax(y_predictions, dim = 1), dim = 1)
                acc_results += (y_predictions_label == y).sum().item() / len(y)
            
            # 4.) Retrun loss and accuracy
            return loss_results / len(data_loader), acc_results / len(data_loader)

    def train_model(self):
        # 1.) Create dictionary for results
        train_results = {
            "train_loss": [],
            "train_acc": [],
            "validation_loss": [],
            "validation_acc": [],
        }

        # 2.) Train model for number of epochs
        for epoch in tqdm(range(self.config.params_epoch)):
            # 2.1) Train step
            train_loss, train_acc = self._train_step()

            # 2.2) Validation step
            validation_loss, validation_acc = self._evaluation_step(data_loader=self.validation_loader)

            # 2.3) Print progress
            if (epoch+1) % 10 == 0:
                print(
                    f"Epoch: {epoch+1} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Train accuracy: {train_acc:.4f} | "
                    f"Validation loss: {validation_loss:.4f} | "
                    f"Validation accuracy: {validation_acc:.4f}"
                )

            # 2.4) Save loss and accuracy to results
            train_results["train_loss"].append(train_loss)
            train_results["train_acc"].append(train_acc)
            train_results["validation_loss"].append(validation_loss)
            train_results["validation_acc"].append(validation_acc)

        # 3.) Return results and trained model
        return train_results
    
    def evaluate_model(self):
        test_loss, test_acc = self._evaluation_step(data_loader=self.test_loader)

        print(f"Test loss: {test_loss:.4f} | "
              f"Test accuracy: {test_acc:.4f} | ")

        test_results = {
            "test_loss": test_loss,
            "test_acc": test_acc
        }
        
        return test_results

    
    def save_model(self, path: Path):
        torch.save(self.model, self.config.trained_model_path)

    @staticmethod
    def plot_results(results):
        pass