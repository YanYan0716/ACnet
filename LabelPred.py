import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers

import config


class LabelPred(keras.layers.Layer):
    def __init__(self, filters, classes):
        super(LabelPred, self).__init__()
        self.net = keras.Sequential([
            keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            keras.layers.Conv2D(
                filters,
                1,
                1,
                kernel_initializer='random_normal',
                kernel_regularizer=config.L2,
                # bias_regularizer=config.L2
            ),
            keras.layers.GlobalAveragePooling2D(),
        ])
        self.l2 = tf.math.l2_normalize
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(
            classes,
            kernel_regularizer=config.L2,
            kernel_initializer='glorot_normal',
            # bias_regularizer=config.L2
            activation='softmax'
        )

    def call(self, inputs, **kwargs):
        out = self.net(inputs)
        out = tf.sign(out) * tf.math.sqrt(tf.sign(out) * out + 1e-12)
        out = self.flatten(self.l2(out, axis=-1))
        out = self.dense(out)
        return out


if __name__ == '__main__':
    img = tf.random.normal((2, 28, 28, 3))
    a = LabelPred(8192, 200)
    y = a(img)
    print(y.shape)
