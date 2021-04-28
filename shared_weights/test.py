"""
    Single modality testing for MISO
"""
import os

from shared_weights.helpers import metrics

from shared_weights.helpers import config, utils
from shared_weights.helpers.siamese_network import create_encoder, create_classifier
from data.data_tf import fat_dataset

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
import numpy as np

from tensorflow.keras.models import model_from_json

model_path = os.environ['PYTHONPATH'].split(os.pathsep)[1] +\
            '/shared_weights/pre_trained_encoders/classifiers/contrastive_resnet50_classifier.json'

weight_path = os.environ['PYTHONPATH'].split(os.pathsep)[1] +\
            '/shared_weights/pre_trained_encoders/classifier_weights/contrastive_resnet50_classifier_weights.h5'


# load json and create model
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weight_path)
print("Loaded model from disk")
print(loaded_model.summary())

new_model = tf.keras.Sequential(loaded_model.layers[1:])

print('New model structure')
print(new_model.summary())

# predict labels for a test set
test_ds = fat_dataset(split='test',
                      data_type='depth',
                      batch_size=config.BATCH_SIZE,
                      shuffle=True,
                      pairs=False)
print('[INFO] loaded FAT dataset')

# evaluate loaded model on test data
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
new_model.compile(loss=metrics.contrastive_loss, optimizer=opt, metrics=['accuracy'])
print('model compiled')

batch_nbr = 0
avg_acc = 0.
print('[INFO] starting prediction task')
for imgs, labels in test_ds:
    # update batch number
    batch_nbr += 1

    # convert labels list to numpy for fast accuracy checking
    labels = np.array(labels)

    score = new_model.evaluate(imgs, labels, verbose=0)
    print("[VALUE] %s: %.2f%%" % (new_model.metrics_names[1], score[1] * 100))

#     # compute batch accuracy
#     acc = np.sum(preds == labels)/len(labels)
#
    # save batch value for final average accuracy
    avg_acc += (score[1] * 100)

#     print(f'[VALUE] Accuracy on batch {batch_nbr}: {round(acc, 4)}')
#
avg_acc /= batch_nbr
print(f'[VALUE] Average accuracy value for {batch_nbr} batches: {round(avg_acc, 2)}')

# imgA = Input(shape=config.IMG_SHAPE)
# imgB = Input(shape=config.IMG_SHAPE)
# featureExtractor = create_encoder(base='resnet101', pretrained=False)
# featsA = featureExtractor(imgA)
# featsB = featureExtractor(imgB)
#
# # finally, construct the siamese network
# distance = Lambda(utils.euclidean_distance)([featsA, featsB])
# outputs = Dense(1, activation="sigmoid")(distance)
# pretrained_encoder = Model(inputs=[imgA, imgB], outputs=outputs)
#
# # load weights into new model
# # pretrained_encoder.load_weights(path_to_encoder + '/shared_weights/pre_trained_encoders/weights/contrastive_resnet50_scratch_weights.h5')
# # pretrained_encoder.load_weights(config.WEIGHT_PATH)
# print("Loaded pre-trained encoder from disk")
#
# pretrained_encoder = tf.keras.Sequential(pretrained_encoder.layers[:-2])
#
# # first train the classification head with a high learning rate
# fine_tuned_classifier = create_classifier(encoder=pretrained_encoder,
#                                           trainable_base=False,
#                                           lr=0.001)
#
# print('Classification Model')
# print(fine_tuned_classifier.summary())

# # load in model weights
# pretrained_encoder.load_weights(base_path)
# print('[INFO] done loading weights')

# classifier = keras.Sequential([
#     keras.layers.Input(shape=config.IMG_SHAPE),
#     pretrained_encoder.layers[1:],
# ])


# # predict labels for a test set
# test_ds = fat_dataset(split='test',
#                       data_type='rgb',
#                       batch_size=config.BATCH_SIZE,
#                       shuffle=True,
#                       pairs=False)
# print('[INFO] loaded FAT dataset')

# batch_nbr = 0
# avg_acc = 0.
# print('[INFO] starting prediction task')
# for imgs, labels in test_ds:
#     # update batch number
#     batch_nbr += 1

#     # convert labels list to numpy for fast accuracy checking
#     labels = np.array(labels)

#     # predict on images
#     preds = classifier.predict(x=imgs, batch_size=config.BATCH_SIZE)

#     # compute batch accuracy
#     acc = np.sum(preds == labels)/len(labels)

#     # save batch value for final average accuracy
#     avg_acc += acc

#     print(f'[VALUE] Accuracy on batch {batch_nbr}: {round(acc, 4)}')

# avg_acc /= batch_nbr
# print(f'[VALUE] Average accuracy value for {batch_nbr} batches: {round(avg_acc, 4)}')
