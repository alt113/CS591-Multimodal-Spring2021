"""
    Contains the siamese network model architecture
"""
# import the necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

input_shape = (28, 28, 3)
dropout_rate = 0.5
hidden_units = 512
num_classes = 10
learning_rate = 0.001


def create_encoder():
    # ResNet50 base encoder
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )
    # specify the inputs for the feature extractor network
    inputs = Input(input_shape)
    # create encodings
    outputs = resnet(inputs)
    # build the model
    model = Model(inputs, outputs, name="supervised_shared_weight_encoder")

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

    model = keras.Model(inputs=inputs, outputs=outputs, name="supervised_shared_weight_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def build_siamese_model(with_top=False):
    if with_top:
        encoder = create_encoder()
        return create_classifier(encoder=encoder)
    else:
        # return the model to the calling function
        return create_encoder()  # no classification top

