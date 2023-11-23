import tensorflow as tf
from tensorflow import keras

# a preprocess for each model

def preprocessForXception(image, label):
  processed_image = tf.image.resize_with_pad(image, 299, 299)
  output_image = keras.applications.xception.preprocess_input(processed_image)
  return output_image, label

# Preprocess the images for preprocessForMobileNetV2
def preprocessForMobileNetV2(image, label):
    image = tf.image.resize(image, [224, 224])  # Resize to 224x224
    image = keras.applications.mobilenet_v2.preprocess_input(image)  # Use MobileNetV2's preprocess_input
    return image, label
