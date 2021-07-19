'''build backbone as the paper said'''
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import tensorflow as tf


def getBackbone(backbone_name):
    if backbone_name == 'RESNET50':
        backbone = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
        )
    else:
        backbone = keras.applications.VGG16(
            include_top=False,
            weights='imagenet'
        )
    return backbone


class Backbone(keras.layers.Layer):
    def __init__(self, backbone):
        super(Backbone, self).__init__()
        self.backbone = getBackbone(backbone)

    def call(self, x, training=False):
        fts = self.backbone(x)
        return fts


if __name__ == '__main__':
    a = getBackbone('RESNET50')
    img = tf.random.normal((2, 448, 448, 3))
