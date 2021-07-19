import tensorflow as tf


from Dataset import dataset
import config


def train():
    ds_train = dataset(config.DATA_PATH)