"""
    Contains the following models to be used as base encoders:

        - LeNet-5
        - AlexNet
        - VGG (16 and 19)
        - Inception
        - ResNet (50, 101, and 152)
"""
from tensorflow import keras


def build_lenet(input_shape=None):
    x = keras.layers.Conv2D(6,
                            kernel_size=5,
                            strides=1,
                            activation='tanh',
                            input_shape=input_shape,
                            padding='same')
    x = keras.layers.AveragePooling2D()(x)
    x = keras.layers.Conv2D(16,
                            kernel_size=5,
                            strides=1,
                            activation='tanh',
                            padding='valid')(x)
    x = keras.layers.AveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    return x


def build_alexnet(input_shape=None):
    x = keras.layers.Conv2D(96,
                            (11, 11),
                            strides=4,
                            input_shape=input_shape)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=2)(x)
    x = keras.layers.Conv2D(256,
                            (5, 5),
                            padding='same')(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=2)(x)
    x = keras.layers.Conv2D(384,
                            (3, 3),
                            padding='same')(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(384,
                            (3, 3),
                            padding='same')(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(256,
                            (3, 3),
                            padding='same')(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=2)(x)
    x = keras.layers.Flatten()(x)

    return x


VGG16 = keras.applications.VGG16
VGG19 = keras.applications.VGG19
INCEPTION = keras.applications.InceptionV3
RESNET50 = keras.applications.ResNet50V2
RESNET101 = keras.applications.ResNet101V2
RESNET152 = keras.applications.ResNet152V2
