import os

from tensorflow.keras import regularizers

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

from ASPP import ASPP
import config
"""from the paper, it is not SelfAttention"""


class Attention(keras.layers.Layer):
    def __init__(self, filters, size):
        super(Attention, self).__init__()
        self.conv_1 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding='same',
            # use_bias=False,
            kernel_initializer=config.CONV_INIT,
            kernel_regularizer=config.L2,
            # bias_regularizer=config.L2
        )
        self.BN_1 = keras.layers.BatchNormalization(
            # epsilon=1e-5,
            # momentum=0.9,
            # trainable=True,
            # scale=True,
            # center=True
            # beta_regularizer=config.L2,
            # gamma_regularizer=config.L2,
        )
        self.relu = keras.layers.ReLU()
        self.conv_2 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding='same',
            # use_bias=False,
            kernel_initializer=config.CONV_INIT,
            kernel_regularizer=config.L2,
            # bias_regularizer=config.L2
        )
        self.BN_2 = keras.layers.BatchNormalization(
            # epsilon=1e-5,
            # momentum=0.9,
            # trainable=True,
            # scale=True,
            # center=True
            # beta_regularizer=config.L2,
            # gamma_regularizer=config.L2,
        )

        self.GAP = keras.layers.GlobalAveragePooling2D()
        self.conv1 = keras.layers.Conv2D(
            filters // 4,
            1,
            1,
            kernel_initializer=config.CONV_INIT,
            activation='relu',
            kernel_regularizer=config.L2,
            # bias_regularizer=config.L2
        )
        
        self.conv2 = keras.layers.Conv2D(
            filters,
            1,
            1,
            kernel_initializer=config.CONV_INIT,
            activation='sigmoid',
            kernel_regularizer=config.L2,
            # bias_regularizer=config.L2
        )
        self.ASPP = ASPP(filters, size)

    def call(self, inputs):
        # img_fts1 = self.ASPP(inputs)
        img_fts1 = self.relu(self.BN_1(self.conv_1(inputs)))
        img_fts1 = self.BN_2(self.conv_2(img_fts1))
        # img_fts1 = self.relu(self.conv_1(inputs))
        # img_fts1 = self.conv_2(img_fts1)

        img_fts2 = self.GAP(img_fts1)
        img_fts2 = tf.expand_dims(img_fts2, axis=1)
        img_fts2 = tf.expand_dims(img_fts2, axis=1)
        img_fts2 = self.conv2(self.conv1(img_fts2))
        out = img_fts1*img_fts2
        out = keras.activations.relu(out)
        return out


if __name__ == '__main__':
    img = tf.random.normal((2, 28, 28, 512))
    a = Attention(512, (28, 28))
    y = a(img)
    print(y.shape)
