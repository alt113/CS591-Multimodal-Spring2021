from multi_input_multi_output.rand_augment import preprocessing_function, HOW_MANY_TO_AUGMENT
import tensorflow as tf
import numpy as np

import os
from PIL import Image
import orjson
import time

data_path = '/projectnb/cs591-mm-ml/prichter/single'

map_labels_dict = {
    '002_master_chef_can_16k': 0,
    '003_cracker_box_16k': 1,
    '004_sugar_box_16k': 2,
    '005_tomato_soup_can_16k': 3,
    '006_mustard_bottle_16k': 4,
    '007_tuna_fish_can_16k': 5,
    '008_pudding_box_16k': 6,
    '009_gelatin_box_16k': 7,
    '010_potted_meat_can_16k': 8,
    '011_banana_16k': 9,
    '019_pitcher_base_16k': 10,
    '021_bleach_cleanser_16k': 11,
    '024_bowl_16k': 12,
    '025_mug_16k': 13,
    '035_power_drill_16k': 14,
    '036_wood_block_16k': 15,
    '037_scissors_16k': 16,
    '040_large_marker_16k': 17,
    '051_large_clamp_16k': 18,
    '052_extra_large_clamp_16k': 19,
    '061_foam_brick_16k': 20,
}


def normalize_image(images, label):
    return tf.cast(images, tf.float32) / 255., label


def augment_image_single(image, label):

    # Create generator and fit it to an image
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    img_gen.fit(image)

    # We want to keep original image and label
    img_results = [image.astype(np.float32)]
    label_results = [label]

    # Perform augmentation and keep the labels
    augmented_images = [next(img_gen.flow(image)) for _ in range(HOW_MANY_TO_AUGMENT)]
    labels = [label for _ in range(HOW_MANY_TO_AUGMENT)]

    # Append augmented data and labels to original data
    img_results.extend(augmented_images)
    label_results.extend(labels)

    return img_results, label_results


def py_augment(image, label):
    """
    In order to use RandAugment inside tf.data.Dataset we must declare a numpy_function
    """
    func = tf.numpy_function(augment_image_single, [image, label], [tf.float32, tf.int32])
    return func


def load_rgb_image(data_path, box, label):
    image = Image.open(data_path).crop(box)
    image = image.resize((128, 128))
    image = tf.cast(np.array(image), tf.int32)
    return (image, label)


def load_depth_image(data_path, box, label):
    image = Image.open(data_path).crop(box)
    image = image.resize((128, 128))
    image = tf.cast(np.array(image), tf.int32)
    image = image * 256 / 65536
    image = tf.expand_dims(image, -1)
    image = tf.broadcast_to(image, (128, 128, 3))
    image = tf.cast(np.array(image), tf.int32)
    return (image, label)


def get_annos(classes, scenes, numbers):
    annos = []
    for c in classes:
        for s in scenes:
            for n in numbers:
                for lr in ['left', 'right']:
                    path = os.path.join(data_path, c, s, n + '.' + lr + '.json')
                    f = open(path)
                    anno = orjson.loads(f.read())
                    if len(anno['objects']) > 0:
                        box = anno['objects'][0]['bounding_box']
                        bbox = [
                            int(box['top_left'][1]),
                            int(box['top_left'][0]),
                            int(box['bottom_right'][1]),
                            int(box['bottom_right'][0])
                        ]
                        annos.append((c, s, n, lr, bbox))
                    f.close()
    return annos


def sample_generator(split='train', data_type='all', batch_size=12, shuffle=False):
    classes = os.listdir(data_path)
    scenes = os.listdir(os.path.join(data_path, classes[0]))
    if split == 'train':
        numbers = [str(im_id).zfill(6) for im_id in range(80)]
    elif split == 'test':
        numbers = [str(im_id).zfill(6) for im_id in range(80, 90)]
    else:
        numbers = [str(im_id).zfill(6) for im_id in range(90, 100)]

    annos = get_annos(classes, scenes, numbers)
    if shuffle:
        np.random.shuffle(annos)

    rgb_paths = []
    depth_paths = []
    boxes = []
    labels = []
    count = 0
    for c, s, n, lr, box in annos:
        rgb_paths.append(os.path.join(data_path, c, s, n + '.' + lr + '.jpg'))
        depth_paths.append(os.path.join(data_path, c, s, n + '.' + lr + '.depth.png'))
        boxes.append(box)
        labels.append(map_labels_dict[c])
        count += 1
        if count == batch_size:
            yield (rgb_paths, depth_paths, boxes, labels)
            rgb_paths = []
            depth_paths = []
            boxes = []
            labels = []
            count = 0
    if count > 0:
        yield (rgb_paths, depth_paths, boxes, labels)


def get_pairs(rgb_paths, depth_paths, boxes, labels, data_type, pairs):
    images = []
    if pairs:
        pair_labels = []
        num_classes = len(map_labels_dict)
        idx = [np.where(labels == i)[0] for i in range(0, num_classes)]
        loaded_images = {}
        for i in range(len(rgb_paths)):
            loaded_images[i] = None
        if data_type == 'rgb':
            paths = rgb_paths
            func = load_rgb_image
        else:
            paths = depth_paths
            func = load_depth_image

        start = tf.timestamp()
        for curr_idx in range(len(rgb_paths)):
            label = labels[curr_idx]
            idx_pos = np.random.choice(idx[label])
            idx_neg = np.random.choice(np.where(labels != label)[0])
            if loaded_images[curr_idx] is None:
                loaded_images[curr_idx], _ = func(paths[curr_idx].numpy(), boxes[curr_idx].numpy(), labels[curr_idx])
            if loaded_images[idx_pos] is None:
                loaded_images[idx_pos], _ = func(paths[idx_pos].numpy(), boxes[idx_pos].numpy(), labels[idx_pos])
            if loaded_images[idx_neg] is None:
                loaded_images[idx_neg], _ = func(paths[idx_neg].numpy(), boxes[idx_neg].numpy(), labels[idx_neg])
            images.append([loaded_images[curr_idx], loaded_images[idx_pos]])
            images.append([loaded_images[curr_idx], loaded_images[idx_neg]])
            pair_labels.append(1)
            pair_labels.append(0)

        return images, tf.cast(np.array(pair_labels), tf.int32)
    else:
        for curr_idx in range(len(rgb_paths)):
            if data_type == 'all':
                rgb_image, _ = load_rgb_image(rgb_paths[curr_idx].numpy(), boxes[curr_idx].numpy(), labels[curr_idx])
                depth_image, _ = load_depth_image(depth_paths[curr_idx].numpy(), boxes[curr_idx].numpy(), labels[curr_idx])
                images.append([rgb_image, depth_image])
            elif data_type == 'rgb':
                rgb_image, _ = load_rgb_image(rgb_paths[curr_idx].numpy(), boxes[curr_idx].numpy(), labels[curr_idx])
                images.append(rgb_image)
            else:
                depth_image, _ = load_depth_image(depth_paths[curr_idx].numpy(), boxes[curr_idx].numpy(), labels[curr_idx])
                images.append(depth_image)
        return images, labels


def fat_dataset(split='train', data_type='all', batch_size=12, shuffle=False, pairs=False):
    assert (pairs == False) or (data_type != 'all')

    ds = tf.data.Dataset.from_generator(
        lambda: sample_generator(split=split, data_type=data_type, batch_size=batch_size, shuffle=shuffle),
        (tf.string, tf.string, tf.int32, tf.int32),
        ([batch_size], [batch_size], [batch_size, 4], [batch_size]),
    )

    ds = ds.map(lambda r, d, b, l: tf.py_function(
        func=get_pairs,
        inp=[r, d, b, l, data_type, pairs],
        Tout=[tf.int32, tf.int32]
    ))

    if data_type != 'all':
        if not pairs:
            ds = ds.map(py_augment).unbatch()
        ds = ds.map(normalize_image)
    return ds


if __name__ == '__main__':
    data_type = 'rgb'
    BATCH_SIZE = 12
    pairs = False
    ds = fat_dataset(data_type=data_type, batch_size=BATCH_SIZE, shuffle=True, pairs=pairs)
    start = time.time()
    for data, labels in ds.take(1):
        print(data.shape)
        print(labels.shape)
        print(time.time() - start)

