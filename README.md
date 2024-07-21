# Dogs_vs_Wolves

## Description

1) **Develop your own end-to-end convolutional neural network in Pytorch. Build a binary classifier that separates wolves from dogs using this kaggle dataset. Using pre-trained CNNs like ResNet, Vgg, DenseNet, etc. is not allowed!**

The convolutional neural network is constructed from 4 convolutional blocks, each consisting of a Convolution layer, ReLU activation, Batch Normalization, MaxPooling, and Dropout. The output layer includes two fully connected layers. The model architecture is primarily inspired by the VGG and ResNet network. It is not modular, so changing the layer order and parameters requires modifications to the code. The model architecture is shown in the image below

<p align="center">
  <img src="/images/cnn_arhitecture.png" />
</p>

2) **Choose your own optimizer, loss function and classification metric to evaluate the performance of your classifier. You can train your network on either GPU (if you possess one) or CPU.**

The chosen loss function is CrossEntropy (BinaryCrossEntropy in this example), and the chosen optimizer is Adam with a learning rate of 0.0001. The learning rate is defined in `param.yaml` and can be changed to any other value. The project pipeline (later explained) also supports training on a GPU. By default, calculations are performed on a CPU, but if a GPU is available, the pipeline switches the model and tensor calculations to the GPU.

3) **Support classification of a single custom image (dog or a wolf) of arbitrary size whose path can be passed via arguments to the main entry point of the project.**

The project supports single image classification with the `customThinker` application and model evaluation on external data. External data must be saved in the `evaluation_data` directory. A toy dataset with 20 images (10 dogs and 10 wolves) for demonstration purposes is already provided.

4) **Evaluate your classifier on test set using appropriate classification metrics. Bonus points for visualizing the classification results.**

The model is trained and tested using a train/validation/test procedure. The train data size is set to 70%, the validation size to 20%, and the test size to 10% (these percentages are chosen arbitrarily; for more information, please refer to this [book](https://link.springer.com/book/10.1007/978-0-387-84858-7)). Since we don't tune hyperparameters during training, the validation set is somewhat unnecessary, but it is included for project demonstration purposes. Further project improvements could focus on defining a tuning procedure. Because we have balanced data, model evaluation during training and testing is performed using classification accuracy and a confusion matrix (conf. mat. only for testing). Training/validation loss, accuracy, and model testing accuracy with a confusion matrix are visualized.

5) **The code should be readable, modular and commented. Add logs for progress tracking during training and evaluation. Include instructions on how to start the training and evaluation of the trained model.**

Logs are added for some project steps. Project building and running are explained in the next section.

**You can use the following hints that will help you to improve classifier and will also bring you bonus points:**
- early stopping, **Done!** :white_check_mark:
- image augmentation (torchvision.transforms), **Done!** :white_check_mark:
- visualize running loss and score (either your own plots ~~or TensorBoar~~), **Done!** :white_check_mark:
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
train_validation_test_config = config.get_train_validation_test_config()

# Prepare data
prepare_data = PrepareData(train_validation_test_config)
train_loader, validation_loader, test_loader = prepare_data.split()

# Initialize model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConvolutionalNeuralNetwork()
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), 
                              lr=train_validation_test_config.params_learning_rate)

# Train model
model.fit(train_validation_test_config.params_epochs,
          train_validation_test_config.params_tolerance,
          train_validation_test_config.params_min_delta,
          train_loader, 
          validation_loader, 
          device, 
          loss_function, 
          optimizer)

test_loss, test_acc, y_pred, y_true  = model.evaluate_model(test_loader, device, loss_function)
```

The `DogVsWolves` package depends on the `ConfigurationManager` (configuration files: `config.yaml` and `params.yaml`). For proper use of the package, the configuration files must be correctly utilized. To avoid exploring the source code, the project includes a built-in short pipeline with two steps:

**step_01_data_ingestion.py:** Downloads data (Dogs vs Wolves - Kaggle dataset) from [Google Drive](https://drive.google.com/file/d/1hyc-VNu-UVPag_FlyL36gBGJBx72VgUf/view?usp=drive_link)

**step_02_train_validate_test_model.py:** Performs data preparation, model initialization, model training/validation, and model testing/evaluation.

The pipeline tracks progress, prints running metrics for model training/testing, saves the model, saves training results (loss and accuracy images), saves test results (confusion matrix image), and saves model parameters. For running the pipeline I recommend the use of DVC (Data Version Control) for pipeline tracking:

```bash
# Initialize dvc
dvc init

# Run pipeline with dvc
dvc repro
```

After the first run, if the pipeline and its dependencies haven't been updated, the data ingestion and training steps won't run again. For each part of the pipeline, we can perform a force run (the second step `step_02_train_validate_test_model` depends on data ingestion and can't run successfully if there is no data):

```bash
# Force run data ingestion
dvc repro --force step_01_data_ingestion

# Force run train/validation/test procedure
dvc repro --force step_02_train_validate_test_model
```

The pipeline can also be run with out dvc (data ingestion and model training is always performed):

```bash
# run project pipeline: data ingestion + model trainig and testing
python main.py
```

Trained models are saved in the artifacts/ directory. The last trained model used for inference is saved in the `model/` directory. The trained model can be evaluated on new data, which must be saved in the `evaluation_data/` directory. One trained model is already saved in the `model/` directory with a toy dataset in the `evaluation_data/` directory (20 images: 10 dogs and 10 wolves), as mentioned before for demonstration purposes. Exceptionally, the training process (train, validation and testing loss/accuracy) of the saved model is shown in the notebook. Trained model evaluation on external data can be performed with:

```bash
# Evaluate last trained model on external data
python model_evaluate_external_data.py
```

Classification of a single custom image (dog or wolf) of arbitrary size can be done by running the `customtkinter` application. Selection and classification of images are performed with buttons. For every image, the predicted class and probability are shown.

```bash
# Run customtkinter app for classification of a single image
python app.py
```

