import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


import config


def readImg(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    return img, label


def augment(img, label):
    img = tf.image.flip_left_right(image=img)
    img = (img / 255.0)
    return img, label


def dataset(dataPath):
    df = pd.read_csv(dataPath)
    file_paths = df['name'].values
    labels = df['label'].values

    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_train = ds_train\
        .map(readImg, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .cache()\
        .shuffle(5000)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)\
        .batch(config.BATCH_SIZE)\


    return ds_train


if __name__ == '__main__':
    ds_train = dataset(config.DATA_PATH)
    for img, label in ds_train:
        print(img.shape)
        print(label)
        plt.imshow(img[0].numpy())
        plt.show()
        break