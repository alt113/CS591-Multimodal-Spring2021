"""
    Holds our implementation of the contrastive_loss function
"""
# import the necessary packages
import tensorflow.keras.backend as K
import tensorflow as tf


def contrastive_loss(y, preds, margin=0.05):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
    
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squared_predictions = K.square(preds)
    squared_margin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squared_predictions + (1 - y) * squared_margin)
    
    # return the computed contrastive loss to the calling function
    return loss
