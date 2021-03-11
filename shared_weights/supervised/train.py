"""
    This script is to pretrain the shared-weight feature encoder network.

    Usage:
    -----
    >> python3 train.py

    *Note: User has to be in the CWD of the train.py script.
"""
# import the necessary packages
from shared_weights.supervised.helpers.siamese_network import build_siamese_model
from shared_weights.supervised.helpers import metrics, config, utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np

# load MNIST dataset and scale the pixel values to the range of [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)

# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model()
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
model = Model(inputs=[imgA, imgB], outputs=distance)

# compile the model
print("[INFO] compiling model...")
model.compile(loss=metrics.contrastive_loss, optimizer="adam")

# train the model
print("[INFO] training model...")
history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
                    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
                    batch_size=config.BATCH_SIZE,
                    epochs=config.EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)

################################################################
# the following code is to make predictions with a trained model
################################################################
# # add channel a dimension to both the images
# imageA = np.expand_dims(imageA, axis=-1)
# imageB = np.expand_dims(imageB, axis=-1)
# # add a batch dimension to both images
# imageA = np.expand_dims(imageA, axis=0)
# imageB = np.expand_dims(imageB, axis=0)
# # scale the pixel values to the range of [0, 1]
# imageA = imageA / 255.0
# imageB = imageB / 255.0
# # use our siamese model to make predictions on the image pair,
# # indicating whether or not the images belong to the same class
# preds = model.predict([imageA, imageB])
# proba = preds[0][0]
