import tensorflow as tf
import numpy as np

import os
from PIL import Image
import orjson

data_path = '/projectnb/cs591-mm-ml/prichter/single'

map_keys = tf.constant([
    '002_master_chef_can_16k',
    '003_cracker_box_16k',
    '004_sugar_box_16k',
    '005_tomato_soup_can_16k',
    '006_mustard_bottle_16k',
    '007_tuna_fish_can_16k',
    '008_pudding_box_16k',
    '009_gelatin_box_16k',
    '010_potted_meat_can_16k',
    '011_banana_16k',
    '019_pitcher_base_16k',
    '021_bleach_cleanser_16k',
    '024_bowl_16k',
    '025_mug_16k',
    '035_power_drill_16k',
    '036_wood_block_16k',
    '037_scissors_16k',
    '040_large_marker_16k',
    '051_large_clamp_16k',
    '052_extra_large_clamp_16k',
    '061_foam_brick_16k',
])

map_values = tf.constant([
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '20',
])

map_labels = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(map_keys, map_values),
    default_value='-1')


def map_class_labels_single(image, label):
    return image, map_labels.lookup(label)


def map_class_labels(rgb, depth, label):
    return rgb, depth, map_labels.lookup(label)


def normalize_image(rgb, depth, label):
    return tf.cast(rgb, tf.float32) / 255., depth, label


def normalize_image_single(image, label):
    return tf.cast(image, tf.float32) / 255., label


def load_image(data_path, box):
    image = Image.open(data_path).crop(box)
    image = image.resize((128, 128))
    image = tf.cast(np.array(image), tf.int32)
    return image


def load_depth_image(data_path, box):
    image = Image.open(data_path).crop(box)
    image = image.resize((128, 128))
    image = tf.cast(np.array(image), tf.int32)
    image = image * 256 / 65536
    image = tf.expand_dims(image, -1)
    image = tf.broadcast_to(image, (128, 128, 3))
    image = tf.cast(np.array(image), tf.int32)
    return image

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

def sample_generator(split='train', data_type='all'):
    classes = os.listdir(data_path)
    scenes = os.listdir(os.path.join(data_path, classes[0]))
    if split == 'train':
        numbers = [str(im_id).zfill(6) for im_id in range(80)]
    elif split == 'test':
        numbers = [str(im_id).zfill(6) for im_id in range(80, 90)]
    else:
        numbers = [str(im_id).zfill(6) for im_id in range(90, 100)]

    annos = get_annos(classes, scenes, numbers)

    for c, s, n, lr, box in annos:
        if data_type == 'all' or data_type == 'rgb':
            rgb_image = load_image(os.path.join(data_path, c, s, n + '.' + lr + '.jpg'), box)
        if data_type == 'all' or data_type == 'depth':
            depth_image = load_depth_image(os.path.join(data_path, c, s, n + '.' + lr + '.depth.png'), box)

        if data_type == 'all':
            yield rgb_image, depth_image, c
        elif data_type == 'rgb':
            yield rgb_image, c
        else:
            yield depth_image, c

def fat_dataset(split='train', data_type='all'):
    out_sig = (tf.int32, tf.int32, tf.string)
    out_shape = ([128, 128, 3], [128, 128, 3], [])
    if data_type != 'all':
        out_sig = (tf.int32, tf.string)
        out_shape = ([128, 128, 3], [])
    ds = tf.data.Dataset.from_generator(
        lambda: sample_generator(split=split, data_type=data_type),
        out_sig,
        out_shape,
    )
    if data_type == 'all':
        ds = ds.map(map_class_labels)
        ds = ds.map(normalize_image)
    else:
        ds = ds.map(map_class_labels_single)
        ds = ds.map(normalize_image_single)
    return ds


ds = fat_dataset(data_type='rgb')

#for image, depth, label in ds.take(1):
#    i = image
#    d = depth
#    l = label
for image, label in ds.take(1):
    i = image
    l = label

print(i.shape)
#print(d.shape)
print(l)
