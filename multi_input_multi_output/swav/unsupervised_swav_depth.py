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
import numpy as np

""" Prepare the data"""
# load FAT dataset
print("[INFO] loading FAT dataset...")
train_ds = fat_dataset(split='train',
                       data_type='depth',
                       batch_size=config.BATCH_SIZE,
                       shuffle=True,
                       pairs=False)

test_ds = fat_dataset(split='test',
                      data_type='depth',
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


def get_projection_prototype(dense_1=1024, dense_2=96, prototype_dimension=10):
    inputs = layers.Input((2048,))
    projection_1 = layers.Dense(dense_1)(inputs)
    projection_1 = layers.BatchNormalization()(projection_1)
    projection_1 = layers.Activation("relu")(projection_1)

    projection_2 = layers.Dense(dense_2)(projection_1)
    projection_2_normalize = tf.math.l2_normalize(projection_2, axis=1, name='projection')

    prototype = layers.Dense(prototype_dimension, use_bias=False, name='prototype')(projection_2_normalize)

    return keras.models.Model(inputs=inputs, outputs=[projection_2_normalize, prototype])


def sinkhorn(sample_prototype_batch):
    Q = tf.transpose(tf.exp(sample_prototype_batch/0.05))
    Q /= tf.keras.backend.sum(Q)
    K, B = Q.shape

    u = tf.zeros_like(K, dtype=tf.float32)
    r = tf.ones_like(K, dtype=tf.float32) / K
    c = tf.ones_like(B, dtype=tf.float32) / B

    for _ in range(3):
        u = tf.keras.backend.sum(Q, axis=1)
        Q *= tf.expand_dims((r / u), axis=1)
        Q *= tf.expand_dims(c / tf.keras.backend.sum(Q, axis=0), 0)

    final_quantity = Q / tf.keras.backend.sum(Q, axis=0, keepdims=True)
    final_quantity = tf.transpose(final_quantity)

    return final_quantity


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
        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.projection_prototype = get_projection_prototype(2048, 128, 63)
        self.obj_for_assign = [0, 1]

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs):
        # Create augmented versions of the images.
        augmented = []
        for _ in range(self.num_augmentations):
            x = data_augmentation(inputs)
            augmented.append(x)
        augmented = layers.Concatenate(axis=0)(augmented)
        # Generate embedding representations of the images.
        features = self.encoder(augmented)
        return features

    def train_step(self, data):#inputs):
        inputs = data[0]
        batch_size = tf.shape(inputs)[0]
        # Run the forward pass and compute the contrastive loss
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            projection, prototype = self.projection_prototype(feature_vectors)
            projection = tf.stop_gradient(projection)
            loss = 0
            for i, obj_id in enumerate(self.obj_for_assign):
                with tape.stop_recording():
                    out = prototype[batch_size * obj_id: batch_size * (obj_id + 1)]
                    q = sinkhorn(out)
                subloss = 0
                for v in np.delete(np.arange(self.num_augmentations), obj_id):
                    p = tf.nn.softmax(prototype[batch_size * v: batch_size * (v + 1)] / self.temperature)
                    subloss -= tf.math.reduce_mean(tf.math.reduce_sum(q * tf.math.log(p), axis=1))
                loss += subloss / self.num_augmentations
            loss = loss / len(self.obj_for_assign)
        # Compute gradients
        trainable_vars = self.encoder.trainable_variables + self.projection_prototype.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update loss tracker metric
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs = data[0]
        batch_size = tf.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        projection, prototype = self.projection_prototype(feature_vectors)
        loss = 0
        for i, obj_id in enumerate(self.obj_for_assign):
            out = prototype[batch_size * obj_id: batch_size * (obj_id + 1)]
            q = sinkhorn(out)
            subloss = 0
            for v in np.delete(np.arange(self.num_augmentations), obj_id):
                p = tf.nn.softmax(prototype[batch_size * v: batch_size * (v + 1)] / self.temperature)
                subloss -= tf.math.reduce_mean(tf.math.reduce_sum(q * tf.math.log(p), axis=1))
            loss += subloss / self.num_augmentations
        loss = loss / len(self.obj_for_assign)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


""" Train the model"""
# Create vision encoder.
network_input = tf.keras.layers.Input(shape=config.IMG_SHAPE)
encoder = create_encoder(base='resnet101', pretrained=False)(network_input)
#encoder_output = tf.keras.layers.Dense(config.HIDDEN_UNITS)(encoder)
#encoder = keras.Model(network_input, encoder_output)
encoder = keras.Model(network_input, encoder)
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

# history = representation_learner.fit(
#     x=train_ds,
#     batch_size=config.BATCH_SIZE,
#     epochs=50,  # for better results, increase the number of epochs to 500.
#     callbacks=[ValidationSinglesAccuracyScore()]
# )

# """ Plot training loss"""

# plt.plot(history["loss"])
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.savefig(config.SINGLE_MODALITY_TRAINING_LOSS_PLOT)

# serialize model to JSON
# serialize weights to HDF5
representation_learner.save_weights(config.DEPTH_MODALITY_WEIGHT_PATH)
print("Saved encoder model to disk")
