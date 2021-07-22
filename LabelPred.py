import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers


class LabelPred(keras.layers.Layer):
    def __init__(self, filters, classes):
        super(LabelPred, self).__init__()
        self.net = keras.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters, 1, 1, activation='relu', kernel_regularizer=regularizers.l2(5e-4)),
            keras.layers.GlobalAveragePooling2D(),
        ])
        # self.l2 = tf.math.l2_normalize
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(classes)

    def call(self, inputs, **kwargs):
        out = self.net(inputs)
        # out = self.flatten(self.l2(out))
        out = self.flatten(out)
        out = self.dense(out)
        return out


if __name__ == '__main__':
    img = tf.random.normal((2, 28, 28, 3))
    a = LabelPred(8192, 200)
    y = a(img)
    print(y.shape)
