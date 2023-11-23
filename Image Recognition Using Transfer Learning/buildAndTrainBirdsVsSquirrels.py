import tensorflow as tf
from tensorflow import keras
from preprocessDefinition import preprocessForMobileNetV2

raw_dataset=tf.data.TFRecordDataset(['/content/drive/MyDrive/Colab_HW/hw3/birds-vs-squirrels-train.tfrecords'])
raw_validation_dataset = tf.data.TFRecordDataset(['/content/drive/MyDrive/Colab_HW/hw3/birds-vs-squirrels-validation.tfrecords'])


def parse_examples(serialized_examples):
  feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                     'label':tf.io.FixedLenFeature([],tf.int64)}
  examples=tf.io.parse_example(serialized_examples,feature_description)
  targets=examples.pop('label')
  images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(
      examples['image'],channels=3),tf.float32),299,299)
  return images, targets

#parse and preprocess
training_set = raw_dataset.map(parse_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)
training_set = training_set.map(preprocessForMobileNetV2, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()


validation_set = raw_validation_dataset.map(parse_examples, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
validation_set = validation_set.map(preprocessForMobileNetV2, num_parallel_calls=tf.data.AUTOTUNE).cache()

base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers on top of the base model
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(512, activation= 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation= 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(3, activation = 'softmax')
])

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

earlyStop_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
optimizer = keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(training_set, validation_data=validation_set, epochs=25, callbacks=[earlyStop_cb])

# Save the model
model.save('/content/drive/MyDrive/Colab_HW/hw3/birdsVsSquirrelsModel.keras')