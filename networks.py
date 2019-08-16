#!/usr/bin/env python2.7
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Sequential


def stacked_network_final_layers(model, output_size, multi_layer_conv_width, pooling):
    while model.layers[-1].output.shape[2] != 1:
        if model.layers[-1].output.shape[2] > 3:
            conv_size = 3
        else:
            conv_size = model.layers[-1].output.shape[2].value
        model.add(Convolution2D(256, (multi_layer_conv_width, conv_size), activation='relu'))
        model.add(Dropout(0.25))

    for n_filters in [256, 512]:
        for ii in range(2):
            model.add(Convolution2D(n_filters, (3, 1), padding='same', activation='relu'))
            model.add(Dropout(0.25))
            if not pooling:
                model.add(Convolution2D(n_filters, (3, 1), strides=(3, 1), activation='relu'))
                model.add(Dropout(0.25))
            else:
                model.add(Convolution2D(n_filters, (3, 1), activation='relu'))
                model.add(Dropout(0.25))
                model.add(MaxPool2D((3, 1)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(output_size, activation='softmax'))

    return model


def mrsnet_small_kernel_no_pool(input_shape, output_size):
    model = Sequential()

    model.add(Convolution2D(256, (7, 1), strides=(2, 1), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Convolution2D(256, (5, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.4))

    model = stacked_network_final_layers(model, output_size, multi_layer_conv_width=3, pooling=False)

    return model


def mrsnet_small_kernel_pool(input_shape, output_size):
    model = Sequential()

    model.add(Convolution2D(256, (7, 1), strides=(2, 1), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Convolution2D(256, (5, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.4))

    model = stacked_network_final_layers(model, output_size, multi_layer_conv_width=3, pooling=True)

    return model


def mrsnet_medium_kernel_no_pool(input_shape, output_size):
    model = Sequential()

    model.add(Convolution2D(256, (9, 1), strides=(2, 1), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Convolution2D(256, (7, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.4))

    model = stacked_network_final_layers(model, output_size, multi_layer_conv_width=5, pooling=False)
    return model


def mrsnet_medium_kernel_pool(input_shape, output_size):
    model = Sequential()

    model.add(Convolution2D(256, (9, 1), strides=(2, 1), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Convolution2D(256, (7, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.4))

    model = stacked_network_final_layers(model, output_size, multi_layer_conv_width=5, pooling=True)
    return model


def mrsnet_large_kernel_no_pool(input_shape, output_size):
    model = Sequential()

    model.add(Convolution2D(256, (16, 1), strides=(2, 1), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Convolution2D(256, (8, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.4))

    model = stacked_network_final_layers(model, output_size, multi_layer_conv_width=7, pooling=False)

    return model


def mrsnet_large_kernel_pool(input_shape, output_size):
    model = Sequential()

    model.add(Convolution2D(256, (16, 1), strides=(2, 1), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Convolution2D(256, (8, 1), strides=(2, 1), activation='relu'))
    model.add(Dropout(0.4))

    model = stacked_network_final_layers(model, output_size, multi_layer_conv_width=7, pooling=True)

    return model
