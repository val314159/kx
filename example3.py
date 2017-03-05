'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs

So that we have 1000 training examples for each class, and 400 validation examples for each class.

In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Input

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
fine_tuned_model_weights_path = 'fine_tuned_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50

# build the VGG16 network
model = Sequential()
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1',
                    input_shape=(img_width, img_height, 3)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1'))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1'))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2'))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool'))

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        TF_WEIGHTS_PATH_NO_TOP,
                        cache_subdir='models')
model.load_weights(weights_path)
#model.load_weights(weights_path)
print('Model1 loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:],name='flatten'))
top_model.add(Dense(256, activation='relu',name='fc1'))
top_model.add(Dropout(0.5,name='dropout1'))
top_model.add(Dense(1, activation='sigmoid',name='binary_prediction'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)
print('Model2 loaded.')

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

if 1:
    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

if 1:
    model.save_weights(fine_tuned_model_weights_path)
