# import the necessary packages
import os

from multi_input_multi_output.models import MultiNet

from shared_weights.helpers import config, utils
from shared_weights.helpers.siamese_network import create_encoder

from data.data_tf import fat_dataset

import tensorflow as tf
from tensorflow import keras

# ----------------------
def flatten_model(model_nested):
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)
    model_flat = keras.models.Sequential(layers_flat)
    return model_flat


""" Data augmentation"""
augmentation_input = keras.layers.Input(shape=config.IMG_SHAPE)
data_augmentation = keras.layers.experimental.preprocessing.RandomTranslation(
    height_factor=(-0.2, 0.2),
    width_factor=(-0.2, 0.2),
    fill_mode="constant"
)(augmentation_input)
data_augmentation = keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal")(data_augmentation)
data_augmentation = keras.layers.experimental.preprocessing.RandomRotation(factor=0.15,
                                                                     fill_mode="constant")(data_augmentation)
augmentation_output = keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.3, 0.1),
                                                                 width_factor=(-0.3, 0.1),
                                                                 fill_mode="constant")(data_augmentation)
data_augmentation = keras.Model(augmentation_input, augmentation_output)

""" Unsupervised contrastive loss"""


class RepresentationLearner(keras.Model):
    def __init__(
        self,
        encoder,
        projection_units,
        num_augmentations,
        temperature=1.0,
        dropout_rate=0.1,
        l2_normalize=False,
        **kwargs
    ):
        super(RepresentationLearner, self).__init__(**kwargs)
        self.encoder = encoder
        # Create projection head.
        self.projector = keras.Sequential(
            [
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(units=projection_units, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
            ]
        )
        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_contrastive_loss(self, feature_vectors, batch_size):
        num_augmentations = tf.shape(feature_vectors)[0] // batch_size
        if self.l2_normalize:
            feature_vectors = tf.math.l2_normalize(feature_vectors, -1)
        # The logits shape is [num_augmentations * batch_size, num_augmentations * batch_size].
        logits = (
            tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
            / self.temperature
        )
        # Apply log-max trick for numerical stability.
        logits_max = tf.math.reduce_max(logits, axis=1)
        logits = logits - logits_max
        # The shape of targets is [num_augmentations * batch_size, num_augmentations * batch_size].
        # targets is a matrix consits of num_augmentations submatrices of shape [batch_size * batch_size].
        # Each [batch_size * batch_size] submatrix is an identity matrix (diagonal entries are ones).
        targets = tf.tile(tf.eye(batch_size), [num_augmentations, num_augmentations])
        # Compute cross entropy loss
        return keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )

    def call(self, inputs):
        # Create augmented versions of the images.
        augmented = []
        for _ in range(self.num_augmentations):
            x = data_augmentation(inputs)
            augmented.append(x)
        augmented = keras.layers.Concatenate(axis=0)(augmented)
        # Generate embedding representations of the images.
        features = self.encoder(augmented)
        # Apply projection head.
        return self.projector(features)

    def train_step(self, data):#inputs):
        inputs = data[0]
        batch_size = tf.shape(inputs)[0]
        # Run the forward pass and compute the contrastive loss
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update loss tracker metric
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):#inputs):
        inputs = data[0]
        batch_size = tf.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


""" Train the model"""
network_input = keras.layers.Input(shape=config.IMG_SHAPE)

# Load RGB vision encoder.
r_encoder = create_encoder(base='resnet50', pretrained=False)(network_input)
encoder_output = keras.layers.Dense(config.HIDDEN_UNITS)(r_encoder)
r_encoder = keras.Model(network_input, encoder_output)
# Create representation learner.
r_representation_learner = RepresentationLearner(
    r_encoder, config.PROJECTION_UNITS, num_augmentations=2, temperature=0.1
)
r_representation_learner.build((None, 128, 128, 3))
# base_path = os.environ['PYTHONPATH'].split(os.pathsep)[1]
# representation_learner.load_weights(base_path + '/multi_input_multi_output/simclr/weights/simclr_resnet50_rgb_scratch_weights.h5')
r_representation_learner.load_weights(config.RGB_MODALITY_WEIGHT_PATH)

functional_model = flatten_model(r_representation_learner.layers[0])
rgb_encoder = functional_model.layers[1]

# Load Depth vision encoder.
d_encoder = create_encoder(base='resnet50', pretrained=False)(network_input)
encoder_output = keras.layers.Dense(config.HIDDEN_UNITS)(d_encoder)
d_encoder = keras.Model(network_input, encoder_output)
# Create representation learner.
d_representation_learner = RepresentationLearner(
    d_encoder, config.PROJECTION_UNITS, num_augmentations=2, temperature=0.1
)
d_representation_learner.build((None, 128, 128, 3))
# base_path = os.environ['PYTHONPATH'].split(os.pathsep)[1]
# representation_learner.load_weights(base_path + '/multi_input_multi_output/simclr/weights/simclr_resnet50_rgb_scratch_weights.h5')
d_representation_learner.load_weights(config.DEPTH_MODALITY_WEIGHT_PATH)

functional_model = flatten_model(d_representation_learner.layers[0])
depth_encoder = functional_model.layers[1]


# ----------------------

# RGB
rgb_input = keras.layers.Input(shape=config.IMG_SHAPE)
# rgb_encoder = keras.applications.ResNet50V2(include_top=False,
#                                             weights=None,
#                                             input_shape=config.IMG_SHAPE,
#                                             pooling="avg")

rgb = rgb_encoder(rgb_input)
rgb = keras.layers.Dropout(config.DROPOUT_RATE)(rgb)
rgb = keras.layers.Dense(config.HIDDEN_UNITS, activation="relu")(rgb)
rgb = keras.layers.Dropout(config.DROPOUT_RATE)(rgb)
rgb = keras.layers.Flatten()(rgb)
rgb = keras.layers.Dense(config.NUM_OF_CLASSES, activation="softmax")(rgb)

rgb_classifier = keras.models.Model(inputs=rgb_input, outputs=rgb, name='rgb_classifier')
for layer in rgb_classifier.layers:
    layer._name += '_rgb'
    layer.trainable = True

print('[INFO] built rgb classifier')
print(rgb_classifier.summary())

# Depth
depth_input = keras.layers.Input(shape=config.IMG_SHAPE)
# depth_encoder = keras.applications.ResNet50V2(include_top=False,
#                                               weights=None,
#                                               input_shape=config.IMG_SHAPE,
#                                               pooling="avg")

depth = depth_encoder(depth_input)
depth = keras.layers.Dropout(config.DROPOUT_RATE)(depth)
depth = keras.layers.Dense(config.HIDDEN_UNITS, activation="relu")(depth)
depth = keras.layers.Dropout(config.DROPOUT_RATE)(depth)
depth = keras.layers.Flatten()(depth)
depth = keras.layers.Dense(config.NUM_OF_CLASSES, activation="softmax")(depth)

depth_classifier = keras.models.Model(inputs=depth_input, outputs=depth, name='depth_classifier')
for layer in depth_classifier.layers:
    layer._name += '_depth'
    layer.trainable = True

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
                                                y={'dense_5_rgb': labels[:], 'dense_7_depth': labels[:]},
                                                reset_metrics=False,
                                                return_dict=True)
        print(f'* Data Batch: {data_batch}')
        print(f'\t{history}')
        break

    if counter % 10 == 0:
        print("[VALUE] Testing model on batch")
        for val_data, val_labels in val_ds:
            val_results = multinet_model.test_on_batch(x=[val_data[:, 0], val_data[:, 1]],
                                                       y={'dense_5_rgb': val_labels[:], 'dense_7_depth': val_labels[:]})
            print(val_results)
            toCSV.append(val_results)

print('Saving MultiNet validation results as CSV file')
utils.save_model_history(H=toCSV, path_to_csv=config.FROZEN_SIAMESE_TRAINING_HISTORY_CSV_PATH)

rgb_classifier.save_weights(config.MIMO_RGB_WEIGHTS)
print("Saved RGB model weights to disk")

# serialize weights to HDF5
depth_classifier.save_weights(config.MIMO_DEPTH_WEIGHTS)
print("Saved Depth model weights to disk")
