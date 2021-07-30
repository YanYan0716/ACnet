import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


import config


def readTrainImg(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (config.SIZE, config.SIZE))
    return img, label


def readTestImg(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = (img / 255.0)
    return img, label


def augment(img, label):
    image = tf.image.random_crop(img, [config.IMG_SIZE, config.IMG_SIZE, 3])
    # image = tf.image.random_brightness(image, max_delta=0.9)
    # image = tf.image.random_contrast(image, lower=0.1, upper=0.9)
    image = tf.image.random_flip_left_right(image)
    image = (image / 255.0)
    return image, label


def dataset(dataPath, train=True):
    df = pd.read_csv(dataPath)
    if train:
        file_paths = df['name'].values
        labels = df['label'].values
        ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        ds_train = ds_train\
            .cache()\
            .shuffle(50000)\
            .map(readTrainImg, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)\
            .batch(config.BATCH_SIZE)
    else:
        file_paths = df['file_name'].values
        labels = df['label'].values
        ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        ds_train = ds_train\
            .cache()\
            .map(readTestImg, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)\
            .batch(1)
    return ds_train


if __name__ == '__main__':
    ds_train = dataset('../CUB_200_2011/aaa.csv')
    for img, label in ds_train:
        print(img.shape)
        print(label)
        plt.imshow(img[0].numpy())
        plt.show()
        break