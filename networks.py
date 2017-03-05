# -*- coding: utf-8 -*-
from pprint import pprint
import os
import h5py
import numpy as np
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D


TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


fine_tuned_model_weights_path = 'fine_tuned_model.h5'
# path to the model weights file.
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50


def network1():
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

    model = Model(img_input, x, name='network1')
    return model


def train1():
    model = network1()

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
    return model


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = mkVGG16()
    print('Model loaded.')
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = mkClassifier(False, train_data.shape[1:])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


def mkVGG16(load_no_top=True):
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
    if load_no_top:
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
        model.load_weights(weights_path)
    return model


def mkClassifier(load_top=True, input_shape=None):
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=input_shape,name='flatten'))
    top_model.add(Dense(256, activation='relu',name='fc1'))
    top_model.add(Dropout(0.5,name='dropout1'))
    top_model.add(Dense(1, activation='sigmoid',name='binary_prediction'))
    if load_top:
        top_model.load_weights(top_model_weights_path)
        print('Model2 loaded.')
        pass
    return top_model


def mkFinetune(load=True):
    model = mkVGG16()
    cls = mkClassifier(False, model.output_shape[1:])
    model.add(cls)
    if load:
        model.load_weights(fine_tuned_model_weights_path)
        pass
    return model


def fine_tune():
    model = mkVGG16()
    top_model = mkClassifier(input_shape=model.output_shape[1:])
    model.add(top_model)
    
    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False
        pass
    
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

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

    model.save_weights(fine_tuned_model_weights_path)
    model.save("model.h5")
    return model
