"""
    Includes helper utilities, including a function to generate image pairs, compute the Euclidean distance as a layer
    inside of a CNN, and a training history plotting function
"""
# import the necessary packages
import csv
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


def make_pairs(images, labels):
    # initialize two empty lists to hold the (rgb, depth) pairs and
    # labels to indicate if a pair is positive or negative
    pair_images = []
    pair_labels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, num_classes)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        curr_img = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idx_b = np.random.choice(idx[label])
        pos_img = images[idx_b]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pair_images.append([curr_img, pos_img])
        pair_labels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        neg_idx = np.where(labels != label)[0]
        neg_img = images[np.random.choice(neg_idx)]
        # prepare a negative pair of images and update our lists
        pair_images.append([curr_img, neg_img])
        pair_labels.append([0])
        # return a 2-tuple of our image pairs and labels
    return np.array(pair_images), np.array(pair_labels)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    
    # compute the sum of squared distances between the vectors
    sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))


def plot_training(H, path_to_plot):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["accuracy"], label="accuracy")
    plt.plot(H["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(path_to_plot)


def save_model_history(H, path_to_csv):
    # save model history dictionary as CSV
    fields = ['loss', 'sparse_categorical_crossentropy']
    with open(path_to_csv, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(H)
