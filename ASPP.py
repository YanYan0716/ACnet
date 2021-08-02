"""reference:https://arxiv.org/pdf/1606.00915.pdf
   https://blog.csdn.net/qq_21997625/article/details/87080576?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.pc_relevant_baidujshouduan&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.pc_relevant_baidujshouduan
"""
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras


import config

class ASPP(keras.layers.Layer):
    def __init__(self, filters, size):
        super(ASPP, self).__init__()
        self.mean = keras.layers.GlobalAveragePooling2D()
        self.conv = keras.layers.Conv2D(
            filters,
            1,
            1,
            kernel_initializer=config.CONV_INIT
        )
        self.upsample = keras.layers.UpSampling2D(size, interpolation='bilinear')

        self.atrous_block1 = keras.layers.Conv2D(
            filters,
            1,
            1,
            kernel_initializer=config.CONV_INIT
        )
        self.atrous_block6 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding='same',
            dilation_rate=6,
            kernel_initializer=config.CONV_INIT
        )
        self.atrous_block12 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding='same',
            dilation_rate=12,
            kernel_initializer=config.CONV_INIT
        )
        self.atrous_block18 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding='same',
            dilation_rate=18,
            kernel_initializer=config.CONV_INIT
        )

        self.conv_1_output = keras.layers.Conv2D(
            filters,
            1,
            1,
            kernel_initializer=config.CONV_INIT
        )

    def call(self, inputs):
        image_features = self.mean(inputs)
        image_features = tf.expand_dims(image_features, axis=1)
        image_features = tf.expand_dims(image_features, axis=1)
        image_features = self.conv(image_features)
        image_features = self.upsample(image_features)

        atrous_block1 = self.atrous_block1(inputs)
        atrous_block6 = self.atrous_block6(inputs)
        atrous_block12 = self.atrous_block12(inputs)
        atrous_block18 = self.atrous_block18(inputs)

        all_fts = tf.concat([
            image_features,
            atrous_block1,
            atrous_block6,
            atrous_block12,
            atrous_block18
        ], axis=-1)
        out = self.conv_1_output(all_fts)
        return out


if __name__ == '__main__':
    img = tf.random.normal((2, 64, 64, 3))
    a = ASPP(5, (64, 64))
    y = a(img)
    print(y.shape)


