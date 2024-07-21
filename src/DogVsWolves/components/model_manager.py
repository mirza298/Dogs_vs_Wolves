import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.io import read_image
from torchvision.transforms import v2

from DogVsWolves.entity.config_entity import TrainValidationTestConfig


def stratified_split(y: np.ndarray, 
                     train_size: float = 0.7, 
                     val_size: float = 0.2, 
                     test_size: float = 0.1, 
                     random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified data split to train set, validation and test set.

    Train, validation and test sizes must sum up to 1: train_size + val_size + test_size === 1.0 !

    Args:
        - y: numpay array with class values (only binary classification supported e.g. [0, 1])
        - train_size: Size of train dataset
        - validation_size: Size of validation dataset
        - test_size: Size of test dataset
        - random_state: Seed for reproducibility

    Return:
        - Tuple: Indices for train, validation and test set

    """
    if random_state is not None:
        np.random.seed(random_state)
    # Indices for each class
    y = np.array(y)
    indices = np.arange(len(y))
    class0_indices = indices[y == 0]
    class1_indices = indices[y == 1]
    
    # Shuffle the indices
    np.random.shuffle(class0_indices)
    np.random.shuffle(class1_indices)
    
    # Calculate the number of samples for each split
    n_train0 = int(train_size * len(class0_indices))
    n_val0 = int(val_size * len(class0_indices))
    n_test0 = len(class0_indices) - n_train0 - n_val0
    
    n_train1 = int(train_size * len(class1_indices))
    n_val1 = int(val_size * len(class1_indices))
    n_test1 = len(class1_indices) - n_train1 - n_val1
    
    # Split indices for class 0
    train_indices0 = class0_indices[:n_train0]
    val_indices0 = class0_indices[n_train0:n_train0+n_val0]
    test_indices0 = class0_indices[n_train0+n_val0:]
    
    # Split indices for class 1
    train_indices1 = class1_indices[:n_train1]
    val_indices1 = class1_indices[n_train1:n_train1+n_val1]
    test_indices1 = class1_indices[n_train1+n_val1:]
    
    # Combine the indices
    train_indices = np.concatenate([train_indices0, train_indices1])
    val_indices = np.concatenate([val_indices0, val_indices1])
    test_indices = np.concatenate([test_indices0, test_indices1])
    
    # Shuffle the combined indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return train_indices, val_indices, test_indices


def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          label_mappings: dict,
                          save_path = None) -> None:
    """
    Plots confusion matrix and prints model accuracy.

    Args:
        - y_true: numpay array of true class labels
        - y_pred: numpay array of predicted class labels
        - label_mappings: dictionary for mapping class labels [0, 1] to names [wolf, dog]
    """

    label_mappings = {v:k for k, v in label_mappings.items()}

    # Define the unique classes
    classes = sorted(set(y_true))

    # Initialize the confusion matrix with zeros
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    # Populate the confusion matrix
    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[true_label][predicted_label] += 1

    # Define class names for better readability
    class_names = [label_mappings[c] for c in classes]

    # Calculate accuracy
    accuracy = confusion_matrix.diagonal().sum() / confusion_matrix.sum()

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Add labels and title
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix for test set, Accuracy: {np.round(accuracy, 4)}')

    # Save plot
    if save_path:
        plt.savefig(str(save_path) + "/test_results.png")

    # Show the plot
    plt.show()


class ImageDataset(Dataset):
    """
    Custom Dataset class for image data.
    """
    def __init__(self, image_directory, transform = None):
        self.image_directory = image_directory
        self.transform = transform
        self.image_labels = os.listdir(image_directory)
        self.image_labels_mapping = {k:v for k, v in zip(self.image_labels, range(len(self.image_labels)))}
        self.image_labels_list = []
        
        # Loop through image_directory and store tuple (image-file-name, image-label) in a list:
        self.image_label_data_pairs = list()
        for path, subdirs, files in os.walk(image_directory):
            for name in files:
                image_path = os.path.join(path, name)
                image_label = self.image_labels_mapping[os.path.basename(path)]
                self.image_labels_list.append(image_label)
                self.image_label_data_pairs.append((image_path, image_label))
        
    def __len__(self):
        return len(self.image_label_data_pairs)
    
    def __getitem__(self, idx):
        path, label = self.image_label_data_pairs[idx]
        image = read_image(path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype = torch.long)
        return image, label
    

class PrepareData:
    """
    Class for data preparation:
        - Augmentation:
             (Resize, RandomHorizontalFlip, RandomVerticalFlip,
             ColorJitter, RandomErasing, Normalize)
        - Stratified data spliting:
            Trainig, Validation and Testing
        - Data loading:
            Create DataLoader instance for each dataset.
    """
    def __init__(self, config: TrainValidationTestConfig):
         self.config = config
    
    def split(self):
        # Data augmentation
        if self.config.params_augmentation:
            self.transform = v2.Compose([
                v2.Resize(
                    size=(self.config.params_image_size, 
                          self.config.params_image_size), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ColorJitter(),
                v2.RandomErasing(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Pytorch standard
            ])
        else:
            self.transform = None

        # Create Dataset instance
        self.data = ImageDataset(image_directory=self.config.data, transform=self.transform)
    
        # Define DataLoader for train(70%), validation(20%) and test(10%) dataset
        train_indices, validation_indices, test_indices = stratified_split(self.data.image_labels_list, 
                                                                           self.config.params_train_size,
                                                                           self.config.params_validation_size,
                                                                           self.config.params_test_size)

        # Define DataLoaders for each dataset
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(validation_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)
        self.train_loader = DataLoader(self.data, batch_size = self.config.params_batch_size, sampler = self.train_sampler)
        self.validation_loader = DataLoader(self.data, batch_size = self.config.params_batch_size, sampler = self.valid_sampler)
        self.test_loader = DataLoader(self.data, batch_size = 1, sampler = self.test_sampler)

        return self.train_loader, self.validation_loader, self.test_loader



class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        """
        CNN model with 4 convolution blocks (Conv layer + Relu activation + Batch Normalization + MaxPolling + Dropout), and two linear layers for output.
        Arhitecture is inspired by VGG network, but much smaller.
        
        ConvolutionalNeuralNetwork ("Tiny VGG"): 
        
            - Number of convolution filters (output channels in pytorch) is equal to 4^(convolution block order) -> 4, 16, 64 and 256.
            - Size of convolution filters (kernel size in pytorch) is equal to 5, for all convolution blocks.
            - Stride for all convolution blocks is equal to 1.
            - Padding is set to 0 for the first and last convolution block, and to 1 for the second and third block.

        Model arhitecture is hardcoded!
        """
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.init_time = time.time()

        # Model Arhitecture
        self.convolution_block = nn.Sequential(
            # First Convolution Layer
            # Input size: [128, 128, 3]
            # Output size: [1+(128 + 2 * padding - kernel_size)/stride,
            #              1+(128 + 2 * padding - kernel_size)/stride,
            #              out_channels]
            # padding: 0, stride: 1, kernel_size: 5
            # Input: 128 x 128 x 3
            # Output: 62 x 62 x 4
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=4),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),

            # Second Convolution Layer
            # Input: 62 x 62 x 4
            # Output: 30 x 30 x 16
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),

            # Third Convolution Layer
            # Input: 30 x 30 x 16
            # Output: 14 x 14 x 64
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),

            # Fourth Convolution Layer
            # Input: 14 x 14 x 64
            # Output: 5 x 5 x 256
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
        )
        
        
        # Linear (FC) Layer - Otuput
        # Input: 5 x 5 x 16
        # Output: 2
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5 * 5 * 256, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features = 2)
        )

        
    def forward(self, x):
        # Convolution Block forward pass
        out = self.convolution_block(x)

        # Output
        out = self.fc_block(out)
        
        return out
    

    def _train_step(self, train_loader, device, loss_function, optimizer):
        # 1.) Set model mode to train
        self.train()
        
        # 2.) Create variables for loss and accuracy
        loss_results, acc_results = 0, 0
        
        # 3.) Loop throught batches
        for i, (X, y) in enumerate(train_loader):
            # 3.1) Move data to device, model has to be on same device!
            X, y = X.to(device), y.to(device)
            
            # 3.2) Forward pass
            y_predictions = self(X)
            
            # 3.3) Calculate loss and accuracy
            loss = loss_function(y_predictions, y)
            loss_results += loss.data.item()
            y_predictions_label = torch.argmax(torch.softmax(y_predictions, dim = 1), dim = 1)
            acc_results += (y_predictions_label == y).sum().item() / len(y)
            
            # 3.4) Empty gradients
            optimizer.zero_grad()
            
            # 3.5) Backward pass
            loss.backward()
            
            # 3.6) Update weights and biases
            optimizer.step()
        
        # 4.) Retrun loss and accuracy
        return loss_results / len(train_loader), acc_results / len(train_loader)
    

    def _evaluation_step(self, evaluation_loader, device, loss_function):
        # 1.) Set model mode to evaluate mode
        self.eval()
        
        # 2.) Create variables for loss and accuracy
        loss_results, acc_results = 0, 0
        predictions, y_true = [], []
        
        with torch.no_grad():
            # 3.) Loop throught batches
            for i, (X, y) in enumerate(evaluation_loader):
                # 3.1) Move data to device, model has to be on same device!
                X, y = X.to(device), y.to(device)

                # 3.2) Forward pass
                pred = self(X)
                
                # 3.3) Calculate loss and accuracy
                loss = loss_function(pred, y)
                loss_results += loss.data.item()
                pred_label = torch.argmax(torch.softmax(pred, dim = 1), dim = 1)
                acc_results += (pred_label == y).sum().item() / len(y)

                predictions = predictions + pred_label.detach().cpu().tolist()
                y_true = y_true + y.detach().cpu().tolist()
            
        # 4.) Retrun loss and accuracy  
        return loss_results / len(evaluation_loader), acc_results / len(evaluation_loader), predictions, y_true
        

    def fit(self, epochs, train_loader, validation_loader, device, loss_function, optimizer):
        """
        Fit CNN model on train data and validate during training.

        Args:
            - train_loader: DataLoader for training data
            - validation_loader: DataLoader for validating data
            - device: Pytorch device (couda or cpu)
            - loss_function: Model loss function
            - optimizer: Model optimizer
        """
        # 1.) Create dictionary for results
        self.train_results = {
            "train_loss": [],
            "train_acc": [],
            "validation_loss": [],
            "validation_acc": [],
        }

        # 2.) Train model for number of epochs
        for epoch in tqdm(range(epochs), file=sys.stdout):
            # 2.1) Train step
            train_loss, train_acc = self._train_step(train_loader, device, loss_function, optimizer)

            # 2.2) Validation step
            validation_loss, validation_acc, _, _ = self._evaluation_step(validation_loader, device, loss_function)

            # 2.3) Print progress
            print('\r\033[2K\033[1G', end='', flush=True)
            if (epoch+1) % 10 == 0:
                print(
                    f"Epoch: {epoch+1} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Train accuracy: {train_acc:.4f} | "
                    f"Validation loss: {validation_loss:.4f} | "
                    f"Validation accuracy: {validation_acc:.4f}"
                )

            # 2.4) Save loss and accuracy to results
            self.train_results["train_loss"].append(train_loss)
            self.train_results["train_acc"].append(train_acc)
            self.train_results["validation_loss"].append(validation_loss)
            self.train_results["validation_acc"].append(validation_acc)
    

    def evaluate_model(self, test_loader, device, loss_function):
        """
        Model evaluation on test data.

        Args:
            - test_loader: DataLoader for test data
            - device: Pytorch device (couda or cpu)
            - loss_function: Model loss function

        Return:
            - Tuple: loss,  accuracy, predictions (binary [0, 1]), True labels
        """
        test_loss, test_acc, predictions, y_true = self._evaluation_step(test_loader, device, loss_function)
        
        return test_loss, test_acc, predictions, y_true

    
    def save_model(self, save_path, trained_model_inference_path):
        """
        Save model to 'model/model.pth' and 'artifacts/trained_model/model_unixtime.pth'
        """
        torch.save(self.state_dict(), str(save_path) +  "/model.pth")
        torch.save(self.state_dict(), trained_model_inference_path)


    def plot_results(self, save_path):
        """
        Plot trainig/validation loss and accuracy. Plots are saved to 'artifacts/trained_model/model_results_unixtime.pth'
        """
        figure, axes = plt.subplots(figsize = (15, 5), ncols = 2)

        axes[0].plot(self.train_results["train_loss"], color = "blue", label = "Train Loss")
        axes[0].plot(self.train_results["validation_loss"], color = "red", label = "Validation Loss")
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(self.train_results["train_acc"], color = "blue", label = "Train Accuracy")
        axes[1].plot(self.train_results["validation_acc"], color = "red", label = "Validation Accuracy")
        axes[1].legend()
        axes[1].grid()

        figure.suptitle(f"Trainig results for model_{self.init_time}")

        figure.savefig(str(save_path) + "/train_results.png")

        plt.show()