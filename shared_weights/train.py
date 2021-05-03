<<<<<<< HEAD
# import the necessary packages
import os

from data.data_tf import fat_dataset
from shared_weights.helpers.siamese_network import create_encoder, create_classifier
from shared_weights.helpers import metrics, config, utils

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json

# # load FAT dataset
# print("[INFO] loading FAT dataset pairs...")
# train_ds = fat_dataset(split='train',
#                       data_type='rgb',
#                       batch_size=config.BATCH_SIZE,
#                       shuffle=True,
#                       pairs=True)

# val_ds = fat_dataset(split='val',
#                      data_type='rgb',
#                      batch_size=12,
#                      shuffle=True,
#                      pairs=True)

# # configure the siamese network
# print("[INFO] building siamese network...")
# imgA = Input(shape=config.IMG_SHAPE)
# imgB = Input(shape=config.IMG_SHAPE)
# featureExtractor = create_encoder(base='resnet101', pretrained=False)
# featsA = featureExtractor(imgA)
# featsB = featureExtractor(imgB)

# # finally, construct the siamese network
# distance = Lambda(utils.euclidean_distance)([featsA, featsB])
# outputs = Dense(1, activation="sigmoid")(distance)
# model = Model(inputs=[imgA, imgB], outputs=outputs)

# # compile the model
# print("[INFO] compiling model...")

# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# loss_option = ['binary_crossentropy', metrics.contrastive_loss]

# model.compile(loss=loss_option[1],
#               optimizer=opt,
#               metrics=['binary_accuracy', 'accuracy'])

# # train the model
# print("[INFO] training encoder...")
# counter = 0
# history = None
# while counter <= config.EPOCHS:
#     counter += 1
#     print(f'* Epoch: {counter}')
#     data_batch = 0
#     for data, labels in train_ds:
#         data_batch += 1
#         history = model.train_on_batch(x=[data[:, 0], data[:, 1]],
#                                       y=labels[:],
#                                       reset_metrics=False,
#                                       return_dict=True)
#         print(f'* Data Batch: {data_batch}')
#         print(f'\t{history}')

#     if counter % 10 == 0:
#         print("[VALUE] Testing model on batch")
#         for val_data, val_labels in val_ds:
#             print(model.test_on_batch(x=[val_data[:, 0], val_data[:, 1]], y=val_labels[:]))

# # serialize model to JSON
# model_json = model.to_json()
# with open(config.MODEL_PATH, "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights(config.WEIGHT_PATH)
# print("Saved encoder model to disk")

##############################################################
# load pre-trained encoder and fine tune classification head #
##############################################################
# load json and create model (python must be version 3.7.7)
# path_to_encoder = os.environ['PYTHONPATH'].split(os.pathsep)[1]
# json_file = open(path_to_encoder + '/shared_weights/pre_trained_encoders/models/contrastive_resnet50_scratch.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# pretrained_encoder = model_from_json(loaded_model_json)

imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = create_encoder(base='resnet101', pretrained=False)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
pretrained_encoder = Model(inputs=[imgA, imgB], outputs=outputs)

# load weights into new model
# pretrained_encoder.load_weights(path_to_encoder + '/shared_weights/pre_trained_encoders/weights/contrastive_resnet50_scratch_weights.h5')
pretrained_encoder.load_weights(config.WEIGHT_PATH)
print("Loaded pre-trained encoder from disk")

pretrained_encoder = Sequential(pretrained_encoder.layers[:-2])

# first train the classification head with a high learning rate
fine_tuned_classifier = create_classifier(encoder=pretrained_encoder,
                                          trainable_base=False,
                                          lr=0.001)

print('Classification Model')
print(fine_tuned_classifier.summary())

train_ds = fat_dataset(split='train',
                      data_type='all',
                      batch_size=config.BATCH_SIZE,
                      shuffle=True,
                      pairs=False)

val_ds = fat_dataset(split='val',
                     data_type='all',
                     batch_size=12,
                     shuffle=True,
                     pairs=False)


# train the model
print("[INFO] training classifier head (frozen base)...")
counter = 0
history = None
toCSV = []
while counter <= 10:
    counter += 1
    print(f'* Epoch: {counter}')
    data_batch = 0
    for data, labels in train_ds:
        data_batch += 1
        history = fine_tuned_classifier.train_on_batch(x=[data[:, 0], data[:, 1]],
                                                      y=labels[:],
                                                      reset_metrics=False,
                                                      return_dict=True)
        print(f'* Data Batch: {data_batch}')
        print(f'\t{history}')

    if counter % 10 == 0:
        print("[VALUE] Testing model on batch")
        for val_data, val_labels in val_ds:
            val_results = fine_tuned_classifier.test_on_batch(x=[val_data[:, 0], val_data[:, 1]], y=val_labels[:])
            print(val_results)
            toCSV.append(val_results)

print('Saving frozen base encoder validation results as CSV file')
# utils.save_model_history(H=toCSV, path_to_csv=path_to_encoder + '/shared_weights/pre_trained_encoders/frozen_history/frozen_cls_base.csv')
utils.save_model_history(H=toCSV, path_to_csv=config.FROZEN_SIAMESE_TRAINING_HISTORY_CSV_PATH)

print('Switching model weights to allow fine tuning of encoder base')
unfrozen_encoder_base = create_classifier(encoder=pretrained_encoder,
                                          trainable_base=True,
                                          lr=1e-5)

unfrozen_encoder_base.set_weights(fine_tuned_classifier.get_weights())

# clear memory of previous model(s)
del fine_tuned_classifier
del pretrained_encoder

# train the model
print("[INFO] training classifier head (unfrozen base)...")
counter = 0
history = None
toCSV = []
while counter <= 10:
    counter += 1
    print(f'* Epoch: {counter}')
    data_batch = 0
    for data, labels in train_ds:
        data_batch += 1
        history = unfrozen_encoder_base.train_on_batch(x=[data[:, 0], data[:, 1]],
                                                      y=labels[:],
                                                      reset_metrics=False,
                                                      return_dict=True)
        print(f'* Data Batch: {data_batch}')
        print(f'\t{history}')

    if counter % 10 == 0:
        print("[VALUE] Testing model on batch")
        for val_data, val_labels in val_ds:
            val_results = unfrozen_encoder_base.test_on_batch(x=[val_data[:, 0], val_data[:, 1]], y=val_labels[:])
            print(val_results)
            toCSV.append(val_results)

print('Saving un-frozen base classifier validation results as CSV file')
# utils.save_model_history(H=toCSV, path_to_csv=path_to_encoder + '/shared_weights/pre_trained_encoders/unfrozen_history/unfrozen_cls_base.csv')
utils.save_model_history(H=toCSV, path_to_csv=config.UNFROZEN_SIAMESE_TRAINING_HISTORY_CSV_PATH)

# serialize model to JSON
model_json = unfrozen_encoder_base.to_json()
# with open(path_to_encoder + '/shared_weights/pre_trained_encoders/classifiers/cls.json', "w") as json_file:
with open(config.FINE_TUNED_CLASSIFICATION_MODEL, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
# unfrozen_encoder_base.save_weights(path_to_encoder + '/shared_weights/pre_trained_encoders/classifier_weights/cls_weights.h5')
unfrozen_encoder_base.save_weights(config.FINE_TUNED_CLASSIFICATION_WEIGHTS)
print("Saved classification model to disk")
