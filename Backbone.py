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


class Backbone(keras.Model):
    def __init__(self, backbone):
        super(Backbone, self).__init__()
        self.backbone = getBackbone(backbone)
        self.bk = keras.Model(inputs=[224, 224, 3], outputs=self.backbone.get_layer('conv4_block4_out').output)

    def call(self, x, training=False):
        y = self.bk(x)
        # fts = self.backbone.get_layer('conv4_block4_out').output
        return y


if __name__ == '__main__':
    a = Backbone('RESNET50')
    img = tf.random.normal((2, 224, 224, 3))
    y = a(img)
    print(y.shape)
