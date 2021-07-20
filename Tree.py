import os

import tensorflow as tf
import tensorflow.keras as keras


from Route import Route
from Attention import Attention
from LabelPred import LabelPred
import config


class BTree(keras.models.Model):
    def __init__(self):
        super(BTree, self).__init__()
