import tensorflow as tf
import numpy as np

import os
from PIL import Image
import orjson

data_path = '/projectnb/cs591-mm-ml/prichter/single'


def map_class_labels(rgb, depth, label):
    if label == '002_master_chef_can_16k':
        return rgb, depth, '0'
    elif label == '003_cracker_box_16k':
        return rgb, depth, '1'
    elif label == '004_sugar_box_16k':
        return rgb, depth, '2'
    elif label == '005_tomato_soup_can_16k':
        return rgb, depth, '3'
    elif label == '006_mustard_bottle_16k':
        return rgb, depth, '4'
    elif label == '007_tuna_fish_can_16k':
        return rgb, depth, '5'
    elif label == '008_pudding_box_16k':
        return rgb, depth, '6'
    elif label == '009_gelatin_box_16k':
        return rgb, depth, '7'
    elif label == '010_potted_meat_can_16k':
        return rgb, depth, '8'
    elif label == '011_banana_16k':
        return rgb, depth, '9'
    elif label == '019_pitcher_base_16k':
        return rgb, depth, '10'
    elif label == '021_bleach_cleanser_16k':
        return rgb, depth, '11'
    elif label == '024_bowl_16k':
        return rgb, depth, '12'
    elif label == '025_mug_16k':
        return rgb, depth, '13'
    elif label == '035_power_drill_16k':
        return rgb, depth, '14'
    elif label == '036_wood_block_16k':
        return rgb, depth, '15'
    elif label == '037_scissors_16k':
        return rgb, depth, '16'
    elif label == '040_large_marker_16k':
        return rgb, depth, '17'
    elif label == '051_large_clamp_16k':
        return rgb, depth, '18'
    elif label == '052_extra_large_clamp_16k':
        return rgb, depth, '19'
    elif label == '061_foam_brick_16k':
        return rgb, depth, '20'


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


def sample_generator(split='train'):
    classes = os.listdir(data_path)
    scenes = os.listdir(os.path.join(data_path, classes[0]))
    if split == 'train':
        numbers = [str(im_id).zfill(6) for im_id in range(80)]
    elif split == 'test':
        numbers = [str(im_id).zfill(6) for im_id in range(80, 90)]
    else:
        numbers = [str(im_id).zfill(6) for im_id in range(90, 100)]

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

    for c, s, n, lr, box in annos:
        rgb_image = load_image(os.path.join(data_path, c, s, n + '.' + lr + '.jpg'), box)
        depth_image = load_depth_image(os.path.join(data_path, c, s, n + '.' + lr + '.depth.png'), box)
        yield rgb_image, depth_image, c


ds = tf.data.Dataset.from_generator(
    sample_generator,
    (tf.int32, tf.int32, tf.string),
    ([128, 128, 3], [128, 128, 3], [])
)

ds = ds.map(map_class_labels)

for image, depth, label in ds.take(1):
    i = image
    d = depth
    l = label

print(i.shape)
print(d.shape)
print(l)
