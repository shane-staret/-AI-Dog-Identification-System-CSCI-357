import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import pathlib

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

'''
Creates paths to the dirctories holding the test and training data
'''
data_dir_train = pathlib.Path("./PhotosTraining")
data_dir_test = pathlib.Path("./PhotosTesting")

# Sets a uniform size for each image to help the dataset
img_height = 180
img_width = 180
# Sets the proper scale for the image values
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# Should be used in main to get the different labels
# image_batch, labels_batch = next(iter(normalized_ds))

def get_trainingData():
    "Splits out the training data and scales the images"
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir_train,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width))
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds
    
def get_validationgData():
    "Splits out the validation data and scales the images"
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir_train,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width))
    normalized_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds

def get_testingData():
    "Reads in the training data and scales the images"
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir_test,
      seed=123,
      image_size=(img_height, img_width))
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds

