import os

from tensorflow.keras import regularizers

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

from ASPP import ASPP

"""from the paper, it is not SelfAttention"""


class Attention(keras.layers.Layer):
    def __init__(self, filters, size):
        super(Attention, self).__init__()
        self.conv_1 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(5e-4),
        )
        self.BN_1 = keras.layers.BatchNormalization(momentum=0.001)
        self.conv_2 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(5e-4),
        )
        self.BN_2 = keras.layers.BatchNormalization(momentum=0.001)

        self.GAP = keras.layers.GlobalAveragePooling2D()
        self.conv1 = keras.layers.Conv2D(
            filters //16,
            1,
            1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(5e-4),
        )
        self.relu = keras.layers.LeakyReLU(alpha=0.2)
        self.conv2 = keras.layers.Conv2D(
            filters,
            1,
            1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(5e-4),
            activation='sigmoid'
        )

        self.ASPP = ASPP(filters, size)

    def call(self, inputs):
        # img_fts1 = self.ASPP(inputs)
        img_fts1 = self.relu(self.BN_1(self.conv_1(inputs)))
        img_fts1 = self.BN_2(self.conv_2(img_fts1))

        img_fts2 = self.GAP(img_fts1)
        img_fts2 = tf.expand_dims(img_fts2, axis=1)
        img_fts2 = tf.expand_dims(img_fts2, axis=1)
        img_fts2 = self.conv2(self.relu(self.conv1(img_fts2)))
        out = tf.einsum('mijn, mpqn -> mijn', img_fts1, img_fts2)
        out = keras.activations.relu(out)
        return out


if __name__ == '__main__':
    img = tf.random.normal((2, 28, 28, 512))
    a = Attention(512, (28, 28))
    y = a(img)
    print(y.shape)
