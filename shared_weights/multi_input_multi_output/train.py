# import the necessary packages
from multi_input_multi_output.models import MultiNet

from shared_weights.helpers import config, utils

from data.data_tf import fat_dataset

from tensorflow import keras

import numpy as np

# RGB
rgb_input = keras.layers.Input(shape=config.IMG_SHAPE)
rgb_encoder = keras.applications.ResNet50V2(include_top=False,
                                            weights=None,
                                            input_shape=config.IMG_SHAPE,
                                            pooling="avg")

rgb = rgb_encoder(rgb_input)
rgb = keras.layers.Dropout(config.DROPOUT_RATE)(rgb)
rgb = keras.layers.Dense(config.HIDDEN_UNITS, activation="relu")(rgb)
rgb = keras.layers.Dropout(config.DROPOUT_RATE)(rgb)
rgb = keras.layers.Flatten()(rgb)
rgb = keras.layers.Dense(config.NUM_OF_CLASSES, activation="softmax")(rgb)

rgb_classifier = keras.models.Model(inputs=rgb_input, outputs=rgb, name='rgb_classifier')
for layer in rgb_classifier.layers:
    layer._name += '_rgb'

print('[INFO] built rgb classifier')
print(rgb_classifier.summary())

# Depth
depth_input = keras.layers.Input(shape=config.IMG_SHAPE)
depth_encoder = keras.applications.ResNet50V2(include_top=False,
                                              weights=None,
                                              input_shape=config.IMG_SHAPE,
                                              pooling="avg")

depth = depth_encoder(depth_input)
depth = keras.layers.Dropout(config.DROPOUT_RATE)(depth)
depth = keras.layers.Dense(config.HIDDEN_UNITS, activation="relu")(depth)
depth = keras.layers.Dropout(config.DROPOUT_RATE)(depth)
depth = keras.layers.Flatten()(depth)
depth = keras.layers.Dense(config.NUM_OF_CLASSES, activation="softmax")(depth)

depth_classifier = keras.models.Model(inputs=depth_input, outputs=depth, name='depth_classifier')
for layer in depth_classifier.layers:
    layer._name += '_depth'

print('[INFO] built depth classifier')
print(depth_classifier.summary())


# Build and compile MultiNet
multinet_class = MultiNet(rgb_classifier=rgb_classifier,
                          rgb_output_branch=rgb,
                          depth_classifier=depth_classifier,
                          depth_output_branch=depth)
multinet_class.compile()

multinet_model = multinet_class.model
print('[INFO] built MultiNet classifier')

# train the network to perform multi-output classification
train_ds = fat_dataset(split='train',
                       data_type='all',
                       batch_size=config.BATCH_SIZE,
                       shuffle=True,
                       pairs=False)

val_ds = fat_dataset(split='validation',
                     data_type='all',
                     batch_size=config.BATCH_SIZE,
                     shuffle=True,
                     pairs=False)

print("[INFO] training MultiNet...")
counter = 0
history = None
toCSV = []
while counter <= config.EPOCHS:
    counter += 1
    print(f'* Epoch: {counter}')
    data_batch = 0
    for imgs, labels in train_ds:
        data_batch += 1
        history = multinet_model.train_on_batch(x=[imgs[:, 0], imgs[:, 1]],
                                                y={'dense_1_rgb': labels[:], 'dense_3_depth': labels[:]},
                                                reset_metrics=False,
                                                return_dict=True)
        print(f'* Data Batch: {data_batch}')
        print(f'\t{history}')
        break

    if counter % 10 == 0:
        print("[VALUE] Testing model on batch")
        for val_data, val_labels in val_ds:
            val_results = multinet_model.test_on_batch(x=[val_data[:, 0], val_data[:, 1]],
                                                       y={'dense_1_rgb': val_labels[:], 'dense_3_depth': val_labels[:]})
            print(val_results)
            toCSV.append(val_results)

print('Saving MultiNet validation results as CSV file')
utils.save_model_history(H=toCSV, path_to_csv=config.FROZEN_SIAMESE_TRAINING_HISTORY_CSV_PATH)

rgb_classifier.save_weights(config.RGB_MODALITY_WEIGHT_PATH)
print("Saved RGB model weights to disk")

# serialize weights to HDF5
depth_classifier.save_weights(config.DEPTH_MODALITY_WEIGHT_PATH)
print("Saved Depth model weights to disk")
