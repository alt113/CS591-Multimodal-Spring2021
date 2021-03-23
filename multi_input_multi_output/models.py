"""
    In order to build our multi-input network we will need two branches:
        - The first branch will be a CNN for the RGB modality
        - The Second branch will also be a CNN but for the depth modality

    These branches will then carry out their own respective classification on the
    input dataset and we will end up having multi-output classification with multi-loss functions
"""
# import the necessary packages
from shared_weights.helpers.siamese_network import create_classifier, create_encoder
from tensorflow.keras.models import Model


def create_rgb_cnn():
    """
        ResNet50 base encoder with a classifier top
    """
    return create_classifier(encoder=create_encoder(), trainable=True)


def create_depth_cnn():
    """
        ResNet50 base encoder with a classifier top
    """
    return create_classifier(encoder=create_encoder(), trainable=True)


def multi_input_multi_output_model():
    """
        Two base encoder with separate classifier tops. This does not perform any
        self-supervised pretraining but instead trains both encoders together from scratch.

        TODO - Check out mix & matching different encoders
    """
    rgb = create_rgb_cnn()
    depth = create_depth_cnn()

    return Model(inputs=[rgb.input, depth.input], outputs=[rgb, depth], name='multinet')
