import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from preprocessDefinition import preprocessForXception

# Load the dataset
trainSet, info = tfds.load(name='oxford_flowers102', split='train+validation', as_supervised=True, with_info=True)
validSet = tfds.load(name='oxford_flowers102', split='test[90%:]', as_supervised=True)
testSet = tfds.load(name='oxford_flowers102', split='test[:90%]', as_supervised=True)

trainSet_processed = trainSet.map(preprocessForXception, num_parallel_calls=32).batch(32).prefetch(1)
validSet_processed = validSet.map(preprocessForXception, num_parallel_calls=32).batch(32).prefetch(1)
testSet_processed = testSet.map(preprocessForXception, num_parallel_calls=32).batch(32).prefetch(1)

#xception model
base_model = keras.applications.xception.Xception(weights = 'imagenet', include_top= False)

avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(info.features['label'].num_classes, activation = 'softmax')(avg)
model = keras.models.Model(inputs = base_model.input, outputs=output)

for layer in base_model.layers:
  layer.trainable = False

#unfreeze some layers for fine tuning
for layer in base_model.layers[-25:]:
  layer.trainable = True

#compile and train model 
model.compile(loss = "sparse_categorical_crossentropy", optimizer = 'adam',
                    metrics=["accuracy"])

model.fit(trainSet_processed, validation_data=validSet_processed, epochs=10,verbose = 2)