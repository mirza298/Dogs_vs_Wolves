# Dogs_vs_Wolves

## Description

1) **Develop your own end-to-end convolutional neural network in Pytorch. Build a binary classifier that separates wolves from dogs using this kaggle dataset. Using pre-trained CNNs like ResNet, Vgg, DenseNet, etc. is not allowed!**

Convolutional neural network is constructed from 4 convolutional blocks (Convolution layer + Relu activation + Batch normalization + MaxPolling + Dropout) and output layer with two fully conected layers. Model arhitecture is mostly inspired form the networks VGG and ResNet, and it isn't modeluar, for different changing layer order and parameters is necessary. Model arhitecture is shown on the image bellow:

<p align="center">
  <img src="/images/cnn_arhitecture.png" />
</p>

2) **Choose your own optimizer, loss function and classification metric to evaluate the performance of your classifier. You can train your network on either GPU (if you possess one) or CPU.**

Chosen loss function is the CrossEntropy (in this example BinaryCrossEntropy), chosen optimizer is Adam with learning rate of 0.0001. Learning rate is a parameter defined in `param.yaml` and can be canged to a any other. Defined CNN model also supports trainig on GPU. By default calculations are performed on CPU, if a GPU is avaible the project switches model and tensor calculations on GPU.

3) **Support classification of a single custom image (dog or a wolf) of arbitrary size whose path can be passed via arguments to the main entry point of the project.**

Project supports single image classification with a `customThinker` application, and model evaluation on external data.

4) **Evaluate your classifier on test set using appropriate classification metrics. Bonus points for visualizing the classification results.**

Model is trained and tested with a train/validation/test procedure. Train data size is set to 70%, validation size to 20% and test size to 10%. Because we have balanced data model evaluation during training (validation) and on testing is performed with classification accuracy. During testing and evaluating on external data, besides classifiaction accuracy we also compute ROC score and the confusion matrix. Training/validation loss, accuracy and model performance metrics (accuracy and ROC score) are visualized.

5) **The code should be readable, modular and commented. Add logs for progress tracking during training and evaluation. Include instructions on how to start the training and evaluation of the trained model.**

Logs are added for some project steps, training tracking and evaluation. Project building and running is explained in the next section.

**You can use the following hints that will help you to improve classifier and will also bring you bonus points:**
- early stopping, **Done!** :white_check_mark:
- image augmentation (torchvision.transforms), **Done!** :white_check_mark:
- visualize running loss and score (either your own plots or TensorBoard), **Done (with own visualizations)!** :white_check_mark:
- data normalization, **Done!** :white_check_mark:
- batch training. **Done!** :white_check_mark:

## How to Build

```bash
# clone project 
git clone https://github.com/mirza298/Dogs_vs_Wolves.git
cd Dogs_vs_Wolves

# create venv
python -m venv /path/to/myenv

# activate venv - bash
source /path/to/myenv/bin/activate

# install project
pip install -r requirements.txt
```

## How to run project

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from DogVsWolves.config.configuration import ConfigurationManager
from DogVsWolves.components.model_manager import *

# Project configuration: data directory, model parameters, ...
config = ConfigurationManager()
model_config = config.get_model_config()

# Prepare data
prepare_data = PrepareData(model_config)
train_loader, validation_loader, test_loader = prepare_data.split()

# Initialize model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConvolutionalNeuralNetwork()
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), 
                            lr=model_config.params_learning_rate)

# Train model
model.fit(model_config.params_epochs, 
          train_loader, 
          validation_loader, 
          device, 
          loss_function, 
          optimizer)

test_loss, test_acc, y_pred, y_true = model.evaluate_model(test_loader, device, loss_function)
```

`DogVsWolves` package depends on the `ConfigurationManager` (configuration files: `config.yaml` and `params.yaml`), for package use configuration files must be correct used. For not exploring the source code, the project has built-in short pipeline with two steps:
**step_01_data_ingestion.py:** Downloads data (Dogs vs Wolves - Kaggle dataset) from [google drive](https://drive.google.com/file/d/1hyc-VNu-UVPag_FlyL36gBGJBx72VgUf/view?usp=drive_link)
**step_02_train_validate_test_model.py:** Performs data preparing, model initialization, model training/validating and model testing/evaluating.

The pipeline performs progress tracking, printing running metrics for model trainig/testing, model saving, saving training results (loss and accuracy image), saving test results (confusion matrix image) and saving model parameters. For running the pipeline use of dvc (Data Version Control) is recommended:

```bash
# Initialize dvc
dvc init

# Run pipeline with dvc
dvc repro
```

After first run, if the pipeline and its dependecies didn't have an update the data ingestion and training won't run again. For each part of pipeline we can perform force run (second step `step_02_train_validate_test_model` depends on data ingestion, it can't run if there is no data):

```bash
# Force run data ingestion
dvc repro --force step_01_data_ingestion

# Force run train/validation/test procedure
dvc repro --force step_02_train_validate_test_model
```

The pipeline can also be run with out dvc (data ingestion and model training is always performed):

```{bash}
# run project pipeline: data ingestion + model trainig and testing
python main.py
```

Trained models are saved into `artifacts/` directory, last trained model used for inference is saved in `model/` directory. Trained model can be evaluated on new data (data must be saved in directory `evaluation_data/`). One trained model is already saved in `model/` directory with a toy dataset in directory `evaluation_data/` (20 images: 10 dogs and 10 wolfs). Trained model evaluation can be performed with:

```{bash}
# Evaluate last trained model on external data
python model_evaluate_external_data.py
```

Classification of a single custom image (dog or a wolf) of arbitrary size can be done by running customtkinter application. Selection and classification of images is performed with buttons. For every image the predicted class and probability is showed.

```{bash}
# Run customtkinter app for classification of a single image
python app.py
```

