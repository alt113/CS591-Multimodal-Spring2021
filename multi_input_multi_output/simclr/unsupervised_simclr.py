# --- START OF SCC/TENSORFLOW CONFIG ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"



import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(os.getenv("CUDA_VISIBLE_DEVICES"))

tf.config.set_soft_device_placement(
    True
)



tf.keras.backend.set_floatx('float32')

# the NSLOTS variable, If NSLOTS is not defined throw an exception.
def get_n_cores():
  nslots = os.getenv('NSLOTS')
  if nslots is not None:
    return int(nslots)
  raise ValueError('Environment variable NSLOTS is not defined.')

tf.config.threading.set_intra_op_parallelism_threads(get_n_cores()-1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# --- END OF SCC/TENSORFLOW CONFIG ---

from data.data_tf import fat_dataset
from shared_weights.helpers import config
from shared_weights.helpers.siamese_network import create_encoder
from shared_weights.helpers.validation_callback import ValidationSinglesAccuracyScore

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

""" Prepare the data"""
# load FAT dataset
print("[INFO] loading FAT dataset...")
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

""" Data augmentation"""
augmentation_input = layers.Input(shape=config.IMG_SHAPE)
data_augmentation = layers.experimental.preprocessing.RandomTranslation(
    height_factor=(-0.2, 0.2),
    width_factor=(-0.2, 0.2),
    fill_mode="constant"
)(augmentation_input)
data_augmentation = layers.experimental.preprocessing.RandomFlip(mode="horizontal")(data_augmentation)
data_augmentation = layers.experimental.preprocessing.RandomRotation(factor=0.15,
                                                                     fill_mode="constant")(data_augmentation)
augmentation_output = layers.experimental.preprocessing.RandomZoom(height_factor=(-0.3, 0.1),
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
                layers.Dropout(dropout_rate),
                layers.Dense(units=projection_units, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
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
        augmented = layers.Concatenate(axis=0)(augmented)
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
# Create vision encoder.
network_input = tf.keras.layers.Input(shape=config.IMG_SHAPE)
encoder = create_encoder(base='resnet50', pretrained=True)(network_input)
encoder_output = tf.keras.layers.Dense(config.HIDDEN_UNITS)(encoder)
encoder = keras.Model(network_input, encoder_output)
# Create representation learner.
representation_learner = RepresentationLearner(
    encoder, config.PROJECTION_UNITS, num_augmentations=2, temperature=0.1
)
# Create a a Cosine decay learning rate scheduler.
lr_scheduler = keras.experimental.CosineDecay(
    initial_learning_rate=0.001, decay_steps=500, alpha=0.1
)
# Compile the model.
representation_learner.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001),
    metrics=['accuracy']
)

# Fit the model
print("[INFO] training encoder...")
counter = 0
history = None
while counter <= config.EPOCHS:
    counter += 1
    print(f'* Epoch: {counter}')
    data_batch = 0
    for data, labels in train_ds:
        data_batch += 1
        history = representation_learner.train_on_batch(x=data[:],
                                                        y=labels[:],
                                                        reset_metrics=False,
                                                        return_dict=True)
        print(f'* Data Batch: {data_batch}')
        print(f'\t{history}')

    if counter % 10 == 0:
        for val_data, val_labels in test_ds:
            print("[VALUE] Testing model on batch")
            print(representation_learner.test_on_batch(x=val_data[:], y=val_labels[:]))

representation_learner.save_weights(config.RGB_MODALITY_WEIGHT_PATH)
print("Saved encoder model to disk")
