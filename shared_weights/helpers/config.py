"""
    Contains our configuration of important variables, including batch size, epochs, output file paths, etc.
"""
# import the necessary packages
import os

# specify the shape of the inputs for our network
IMG_SHAPE = (128, 128, 3)

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 130

# specify the projection head size
PROJECTION_UNITS = 128

# number of classes
NUM_OF_CLASSES = 21

# number of hidden units
HIDDEN_UNITS = 512

# Dropout rate
DROPOUT_RATE = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_plot.png"])

RGB_MODALITY_PATH = os.path.sep.join([BASE_OUTPUT, "rgb_modality_model"])
