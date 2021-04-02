"""
    This script is to pretrain the shared-weight feature encoder network.

    Usage:
    -----
    >> python3 train.py

    *Note: User has to be in the CWD of the train.py script.
"""
# import the necessary packages
from data.data_tf import fat_dataset
from shared_weights.helpers.siamese_network import create_encoder
from shared_weights.helpers import metrics, config, utils

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense

# load FAT dataset
print("[INFO] loading FAT dataset pairs...")
train_ds = fat_dataset(split='train',
                       data_type='rgb',
                       batch_size=config.BATCH_SIZE,
                       shuffle=True,
                       pairs=True)

val_ds = fat_dataset(split='val',
                     data_type='rgb',
                     batch_size=12,
                     shuffle=True,
                     pairs=True)

# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = create_encoder(base='vgg19', pretrained=False)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
print("[INFO] compiling model...")

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_option = ['binary_crossentropy', metrics.contrastive_loss]

model.compile(loss=loss_option[1],
              optimizer=opt,
              metrics=['binary_accuracy', 'accuracy'])

# train the model
print("[INFO] training encoder...")
counter = 0
history = None
while counter <= config.EPOCHS:
    counter += 1
    print(f'* Epoch: {counter}')
    data_batch = 0
    for data, labels in train_ds:
        data_batch += 1
        history = model.train_on_batch(x=[data[:, 0], data[:, 1]],
                                       y=labels[:],
                                       reset_metrics=False,
                                       return_dict=True)
        print(f'* Data Batch: {data_batch}')
        print(f'\t{history}')

    if counter % 10 == 0:
        for val_data, val_labels in val_ds:
            print("[VALUE] Testing model on batch")
            print(model.test_on_batch(x=[val_data[:, 0], val_data[:, 1]], y=val_labels[:]))

# serialize model to JSON
model_json = model.to_json()
with open(config.MODEL_PATH, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(config.WEIGHT_PATH)
print("Saved encoder model to disk")

# plot the training history
print("[INFO] saving training history as CSV...")
utils.save_model_history(history, config.SIAMESE_TRAINING_HISTORY_CSV_PATH)

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
