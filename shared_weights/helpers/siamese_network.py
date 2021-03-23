"""
    Contains the siamese network model architecture
"""
# import the necessary packages
from shared_weights.helpers.base_encoders import RESNET50, RESNET101, RESNET152, INCEPTION
from shared_weights.helpers.base_encoders import build_lenet, build_alexnet, VGG16, VGG19

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

input_shape = (128, 128, 3)
dropout_rate = 0.5
hidden_units = 512
num_classes = 10
learning_rate = 0.001


def create_encoder(base='resnet50'):
    base_encoder = None
    if base == 'resnet50':
        base_encoder = RESNET50(
            include_top=False, weights=None, input_shape=input_shape, pooling="avg"
        )
    elif base == 'resnet101':
        base_encoder = RESNET101(
            include_top=False, weights=None, input_shape=input_shape, pooling="avg"
        )
    elif base == 'resnet152':
        base_encoder = RESNET152(
            include_top=False, weights=None, input_shape=input_shape, pooling="avg"
        )
    elif base == 'inception':
        base_encoder = INCEPTION(
            include_top=False, weights=None, input_shape=input_shape, pooling="avg"
        )
    elif base == 'lenet':
        base_encoder = build_lenet(input_shape=input_shape)
    elif base == 'alexnet':
        base_encoder = build_alexnet(input_shape=input_shape)
    elif base == 'vgg16':
        base_encoder = VGG16(
            include_top=False, weights=None, input_shape=input_shape, pooling="avg"
        )
    elif base == 'vgg19':
        base_encoder = VGG19(
            include_top=False, weights=None, input_shape=input_shape, pooling="avg"
        )

    # specify the inputs for the feature extractor network
    inputs = Input(input_shape)
    # create encodings
    outputs = base_encoder(inputs)
    # build the model
    model = Model(inputs, outputs, name="base_encoder")

    # return the model to the calling function
    return model


def create_classifier(encoder, trainable=False):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
