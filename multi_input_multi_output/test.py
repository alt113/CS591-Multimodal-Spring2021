"""
    Single modality testing for MIMO
"""
import os
import argparse

from shared_weights.helpers import config
from data.data_tf import fat_dataset

from tensorflow import keras
import numpy as np

base_path = os.environ['PYTHONPATH'].split(os.pathsep)[1] + '/multi_input_multi_output/simclr/weights'

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-m", "--Modality", help="Target Modality")

# Read arguments from command line
args = parser.parse_args()

model_weight_path = None
if str(args.Modality).lower() == 'rgb':
    # model_weight_path = config.MIMO_RGB_WEIGHTS
    model_weight_path = base_path + '/mimo_rgb_imagenet_weights.h5'
elif str(args.Modality).lower() == 'depth':
    # model_weight_path = config.MIMO_DEPTH_WEIGHTS
    model_weight_path = base_path + '/mimo_depth_imagenet_weights.h5'
else:
    raise Exception('Only RGB and Depth modalities are accepted.')

network_input = keras.layers.Input(shape=config.IMG_SHAPE)
feature_encoder = keras.applications.ResNet50V2(include_top=False,
                                                weights=None,
                                                input_shape=config.IMG_SHAPE,
                                                pooling="avg")

x = feature_encoder(network_input)
x = keras.layers.Dropout(config.DROPOUT_RATE)(x)
x = keras.layers.Dense(config.HIDDEN_UNITS, activation="relu")(x)
x = keras.layers.Dropout(config.DROPOUT_RATE)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(config.NUM_OF_CLASSES, activation="softmax")(x)

classifier = keras.models.Model(inputs=network_input, outputs=x, name='single_modality_classifier')
print('[INFO] done building saved classifier')

# load in model weights
classifier.load_weights(model_weight_path)
print('[INFO] done loading weights')

# predict labels for a test set
test_ds = fat_dataset(split='test',
                      data_type=str(args.Modality).lower(),
                      batch_size=config.BATCH_SIZE,
                      shuffle=True,
                      pairs=False)
print('[INFO] loaded FAT dataset')

batch_nbr = 0
avg_acc = 0.
k = 5
print('[INFO] starting prediction task')
for imgs, labels in test_ds:
    # update batch number
    batch_nbr += 1

    # convert labels list to numpy for fast accuracy checking
    labels = np.array(labels)

    # predict on images
    probs = classifier.predict(x=imgs, batch_size=config.BATCH_SIZE)
    # preds = np.argmax(probs, axis=1)

    # do top-k accuracy calculation
    preds = probs.argsort(axis=1)[-k:][::-1]

    nbr_correct = 0
    for p, gt in zip(preds, labels):
        if gt in p:
            nbr_correct += 1

    acc = nbr_correct/len(labels)


    # # compute batch accuracy
    # acc = np.sum(preds == labels)/len(labels)

    # save batch value for final average accuracy
    avg_acc += acc

    print(f'[VALUE] Top-{k}  Accuracy on batch {batch_nbr}: {round(acc, 4)}')

avg_acc /= batch_nbr
print(f'[VALUE] Average accuracy value for {batch_nbr} batches: {round(avg_acc, 4)}')
