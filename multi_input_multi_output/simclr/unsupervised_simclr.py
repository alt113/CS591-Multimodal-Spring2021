from data.data_tf import fat_dataset
from shared_weights.helpers import config
from shared_weights.helpers.siamese_network import create_encoder

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

""" Prepare the data"""
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

""" Define hyperparameters"""

target_size = 32  # Resize the input images.

"""# Implement data preprocessing"""

data_preprocessing = keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(target_size, target_size),
        layers.experimental.preprocessing.Normalization(),
    ]
)

""" Data augmentation"""

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomTranslation(
            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"
        ),
        layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
        layers.experimental.preprocessing.RandomRotation(
            factor=0.15, fill_mode="nearest"
        ),
        layers.experimental.preprocessing.RandomZoom(
            height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"
        )
    ]
)

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
        # Preprocess the input images.
        preprocessed = data_preprocessing(inputs)
        # Create augmented versions of the images.
        augmented = []
        for _ in range(self.num_augmentations):
            augmented.append(data_augmentation(preprocessed))
        augmented = layers.Concatenate(axis=0)(augmented)
        # Generate embedding representations of the images.
        features = self.encoder(augmented)
        # Apply projection head.
        return self.projector(features)

    def train_step(self, inputs):
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

    def test_step(self, inputs):
        batch_size = tf.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


""" Train the model"""
# Create vision encoder.
encoder = create_encoder(base='resnet50')
encoder = tf.keras.layers.Dense(config.HIDDEN_UNITS)(encoder)
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

# history = representation_learner.fit(
#     x=x_data,
#     batch_size=512,
#     epochs=50,  # for better results, increase the number of epochs to 500.
# )

""" Plot training loss"""
#
# plt.plot(history.history["loss"])
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.show()
