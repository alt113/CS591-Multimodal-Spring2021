from data.data_tf import fat_dataset
from shared_weights.helpers import config

from tensorflow import keras
import numpy as np

singles_val_ds = fat_dataset(split='val',
                             data_type='rgb',
                             batch_size=12,
                             shuffle=True,
                             pairs=False)


class ValidationSinglesAccuracyScore(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for data_singles, labels in singles_val_ds.take(1):
            predictions = self.model.predict_on_batch(data_singles)
            val_acc = np.sum(predictions.ravel() == labels)/config.BATCH_SIZE
            print(f'[EVALUATION] Epoch: {epoch}, Validation Accuracy: {round(val_acc, 4)}')
