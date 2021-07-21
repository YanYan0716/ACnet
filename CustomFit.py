import tensorflow as tf
import tensorflow.keras as keras


class CustomFit(keras.Model):
    def __init__(self, model):
        super(CustomFit, self).__init__()

    def call(self, inputs, training=None, mask=None):
        pass