# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

x = img_input = Input(shape=(150, 150, 3))

# Block 1
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

# Classification block
x = Flatten(name='flatten')(x)
x = Dense(64, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid', name='binary_predictions')(x)

model = Model(img_input, x, name='vgg16')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)

model.save_weights('first_try.h5')  # always save your weights after training or during training
