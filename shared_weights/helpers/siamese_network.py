"""
    Contains the siamese network model architecture
"""
# import the necessary packages
from shared_weights.helpers.base_encoders import RESNET50, RESNET101, RESNET152, INCEPTION
from shared_weights.helpers.base_encoders import build_lenet, build_alexnet, VGG16, VGG19
from shared_weights.helpers import config

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def create_encoder(base='resnet50', pretrained=False):
    base_encoder = None
    weights = 'imagenet' if pretrained else None

    if base == 'resnet50':
        base_encoder = RESNET50(
            include_top=False, weights=weights, input_shape=config.IMG_SHAPE, pooling="avg"
        )
    elif base == 'resnet101':
        base_encoder = RESNET101(
            include_top=False, weights=weights, input_shape=config.IMG_SHAPE, pooling="avg"
        )
    elif base == 'resnet152':
        base_encoder = RESNET152(
            include_top=False, weights=weights, input_shape=config.IMG_SHAPE, pooling="avg"
        )
    elif base == 'inception':
        base_encoder = INCEPTION(
            include_top=False, weights=weights, input_shape=config.IMG_SHAPE, pooling="avg"
        )
    elif base == 'lenet':
        base_encoder = build_lenet(input_shape=config.IMG_SHAPE)
    elif base == 'alexnet':
        base_encoder = build_alexnet(input_shape=config.IMG_SHAPE)
    elif base == 'vgg16':
        base_encoder = VGG16(
            include_top=False, weights=weights, input_shape=config.IMG_SHAPE, pooling="avg"
        )
    elif base == 'vgg19':
        base_encoder = VGG19(
            include_top=False, weights=weights, input_shape=config.IMG_SHAPE, pooling="avg"
        )

    # specify the inputs for the feature extractor network
    inputs = Input(config.IMG_SHAPE)
    # create encodings
    outputs = base_encoder(inputs)
    # build the model
    model = Model(inputs, outputs, name="base_encoder")

    # return the model to the calling function
    return model


def create_classifier(encoder, trainable_base=False, lr=0.001):

    for layer in encoder.layers:
        layer.trainable = trainable_base

    encoder.add(layers.Dropout(config.DROPOUT_RATE))
    encoder.add(layers.Dense(config.HIDDEN_UNITS, activation="relu"))
    encoder.add(layers.Dropout(config.DROPOUT_RATE))
    encoder.add(layers.Dense(config.NUM_OF_CLASSES, activation="softmax"))

    # inputs = keras.Input(shape=config.IMG_SHAPE)
    # features = encoder(inputs)
    # features = layers.Dropout(config.DROPOUT_RATE)(features)
    # features = layers.Dense(config.HIDDEN_UNITS, activation="relu")(features)
    # features = layers.Dropout(config.DROPOUT_RATE)(features)
    # outputs = layers.Dense(config.NUM_OF_CLASSES, activation="softmax")(features)
    #
    # model = keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    encoder.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    return encoder
