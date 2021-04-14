"""
    Contains our configuration of important variables, including batch size, epochs, output file paths, etc.
"""
# import the necessary packages
import os

# specify the shape of the inputs for our network
IMG_SHAPE = (128, 128, 3)

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 50

# specify the projection head size
PROJECTION_UNITS = 128

# number of classes
NUM_OF_CLASSES = 21

# number of hidden units
HIDDEN_UNITS = 512

# Dropout rate
DROPOUT_RATE = 0.5

# define the path to the base output directory
BASE_OUTPUT = "shared_weights/pre_trained_encoders"
MULTINET_BASE_OUTPUT = 'multi_input_multi_output/simclr'

# use the base output path to derive the path to the serialized
# model along with training history plot
# -- Shared Weights --
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "models/contrastive_resnet101_scratch.json"])
BCE_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "models/bce_resnet101_scratch.json"])

FINE_TUNED_CLASSIFICATION_MODEL = os.path.sep.join([BASE_OUTPUT, "classifiers/contrastive_resnet101_classifier.json"])
BCE_FINE_TUNED_CLASSIFICATION_MODEL = os.path.sep.join([BASE_OUTPUT, "classifiers/bce_resnet101_classifier.json"])

WEIGHT_PATH = os.path.sep.join([BASE_OUTPUT, "weights/contrastive_resnet101_scratch_weights.h5"])
BCE_WEIGHT_PATH = os.path.sep.join([BASE_OUTPUT, "weights/bce_resnet101_scratch_weights.h5"])
FINE_TUNED_CLASSIFICATION_WEIGHTS = os.path.sep.join([BASE_OUTPUT, "classifier_weights/contrastive_resnet101_classifier_weights.h5"])
BCE_FINE_TUNED_CLASSIFICATION_WEIGHTS = os.path.sep.join([BASE_OUTPUT, "classifier_weights/bce_resnet101_classifier_weights.h5"])

PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_plot.png"])
FROZEN_SIAMESE_TRAINING_HISTORY_CSV_PATH = os.path.sep.join([BASE_OUTPUT, "frozen_history/contrastive_resnet101_classifier_history.csv"])
UNFROZEN_SIAMESE_TRAINING_HISTORY_CSV_PATH = os.path.sep.join([BASE_OUTPUT,
                                                               "unfrozen_history/contrastive_resnet101_classifier_history.csv"])
BCE_FROZEN_SIAMESE_TRAINING_HISTORY_CSV_PATH = os.path.sep.join([BASE_OUTPUT, "frozen_history/bce_resnet101_classifier_history.csv"])
BCE_UNFROZEN_SIAMESE_TRAINING_HISTORY_CSV_PATH = os.path.sep.join([BASE_OUTPUT,
                                                               "unfrozen_history/bce_resnet101_classifier_history.csv"])


# -- Multinet Single Modalities --
RGB_MODALITY_MODEL_PATH = os.path.sep.join([MULTINET_BASE_OUTPUT, "models/simclr_resnet50_rgb_scratch.json"])
RGB_MODALITY_WEIGHT_PATH = os.path.sep.join([MULTINET_BASE_OUTPUT, "weights/simclr_resnet50_rgb_scratch_weights.h5"])

DEPTH_MODALITY_MODEL_PATH = os.path.sep.join([MULTINET_BASE_OUTPUT, "models/simclr_resnet50_depth_scratch.json"])
DEPTH_MODALITY_WEIGHT_PATH = os.path.sep.join([MULTINET_BASE_OUTPUT, "weights/simclr_resnet50_depth_scratch_weights.h5"])

# save the loss plot after training for evaluation
SINGLE_MODALITY_TRAINING_LOSS_PLOT = os.path.sep.join([MULTINET_BASE_OUTPUT,
                                                       "loss_plots/resnet50_rgb.png"])
