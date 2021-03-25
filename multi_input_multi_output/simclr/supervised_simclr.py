from data.data_tf import fat_dataset
from shared_weights.helpers import config
from shared_weights.helpers.siamese_network import create_encoder, create_classifier

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

""" Constants and Hyperparameters """
learning_rate = 0.001
temperature = 0.05

""" Data Preparation Section"""
# load FAT dataset
print("[INFO] loading FAT dataset pairs...")
train_ds = fat_dataset(split='train',
                       data_type='rgb',
                       batch_size=config.BATCH_SIZE,
                       shuffle=True,
                       pairs=False)

test_ds = fat_dataset(split='test',
                      data_type='rgb',
                      batch_size=config.BATCH_SIZE,
                      shuffle=True,
                      pairs=False)

"""# Data Augmentation"""

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.02),
        layers.experimental.preprocessing.RandomWidth(0.2),
        layers.experimental.preprocessing.RandomHeight(0.2),
    ]
)

"""# Build the encoder model"""
encoder = create_encoder(base='resnet50')
inputs = keras.Input(shape=config.IMG_SHAPE)
augmented = data_augmentation(inputs)
outputs = encoder(augmented)
model = keras.Model(inputs=inputs, outputs=outputs, name="rgb_encoder")

print(model.summary())


""" Supervised contrastive learning loss function """


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1., name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder):
    inputs = keras.Input(shape=config.IMG_SHAPE)
    features = encoder(inputs)
    outputs = layers.Dense(config.PROJECTION_UNITS, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="simplenet-encoder_with_projection-head"
    )
    return model


""" Pretrain the encoder"""
encoder_with_projection_head = add_projection_head(model)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

print(encoder_with_projection_head.summary())

print("[INFO] training encoder...")
counter = 0
history = None
while counter <= config.EPOCHS:
    counter += 1
    print(f'* Epoch: {counter}')
    data_batch = 0
    for data, labels in train_ds:
        data_batch += 1
        history = encoder_with_projection_head.train_on_batch(x=data[:],
                                                              y=labels[:],
                                                              reset_metrics=False,
                                                              return_dict=True)
        print(f'* Data Batch: {data_batch}')
        print(f'\t{history}')

    if counter % 10 == 0:
        for val_data, val_labels in test_ds:
            print("[VALUE] Testing model on batch")
            print(encoder_with_projection_head.test_on_batch(x=val_data[:], y=val_labels[:]))


# serialize model to JSON
model_json = model.to_json()
with open(config.RGB_MODALITY_PATH, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("rgb_modality_model_weights.h5")
print("Saved model to disk")

# TODO - do we want to train classifier heads separately as well or together?
# """ Train the classifier with the frozen encoder """
#
# classifier = create_classifier(model, trainable_base=False)
#
# print("[INFO] training classifier head...")
# counter = 0
# history = None
# while counter <= config.EPOCHS:
#     counter += 1
#     print(f'* Epoch: {counter}')
#     data_batch = 0
#     for data, labels in train_ds:
#         data_batch += 1
#         history = classifier.train_on_batch(x=data[:],
#                                             y=labels[:],
#                                             reset_metrics=False,
#                                             return_dict=True)
#         print(f'* Data Batch: {data_batch}')
#         print(f'\t{history}')
#
#     if counter % 10 == 0:
#         for val_data, val_labels in test_ds:
#             print("[VALUE] Testing model on batch")
#             print(classifier.test_on_batch(x=val_data[:], y=val_labels[:]))
