# Image Classification Using Transfer Learning

## Overview

This repository contains the implementation of two deep learning models aimed at classifying images of flowers and distinguishing between birds and squirrels. The models are built using TensorFlow and Keras and employ transfer learning for efficient training.

## Project Structure

preprocessDefinition.py - Contains preprocessing functions for both models. Functions include resizing images, applying Xception's preprocessing, and formatting the data for model input.

buildAndTrainFlowersModel.py - Script for training the flower classification model on the Oxford Flowers 102 dataset using the Xception architecture.

buildAndTrainBirdsVsSquirrels.py - Script for training a model to classify between birds and squirrels/chipmunks using the MobileNetV2 architecture.

flowersModel.keras.zip - Compressed file containing the trained flower classification model.

birdsVsSquirrelsModel.keras - Saved model file for the birds vs. squirrels classifier.

## Models

### Flowers Model

Utilizes the Xception model pre-trained on ImageNet. The script buildAndTrainFlowersModel.py performs the following steps:

Loads the Oxford Flowers 102 dataset and splits it into training, validation, and testing sets.

Applies preprocessing suitable for the Xception model.

Modifies the Xception architecture to fit our specific class count (102 classes).

Trains the model with fine-tuning on the last 25 layers of the base network.

### Birds vs. Squirrels Classifier
Employs the MobileNetV2 model pre-trained on ImageNet. The script buildAndTrainFlowersModel.py performs the following steps:

Loading a custom training and validation dataset and applying suitable preprocessing.

Modifies the MobileNetV2 architecture to fit our specific class count (3 classes).

Compiling and training the model, including callback definitions for monitoring performance.

## Usage

### Preprocessing
Run preprocessDefinition.py to preprocess images before feeding them into the models. Make sure to adjust the script to point to your dataset location.

### Training Models
Execute buildAndTrainFlowersModel.py and buildAndTrainBirdsVsSquirrels.py to train the respective models. Training parameters such as epochs, batch size, and validation splits are set within these scripts and can be adjusted as needed.

### Model Evaluation
After training, the models can be evaluated using the test splits of the datasets to measure accuracy and loss metrics. The evaluation can be performed by adding code for model.evaluate at the end of the training scripts.

## Requirements

TensorFlow 2.x
TensorFlow Datasets
NumPy
Matplotlib
Installation

## To set up the required environment:

sh
Copy code
pip install tensorflow tensorflow_datasets numpy matplotlib
Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
