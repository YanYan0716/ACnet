import tensorflow as tf
import tensorflow.keras as keras
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
from ASPP import ASPP

"""from the paper, it is not SelfAttention"""


class Attention(keras.layers.Layer):
    def __init__(self, filters, size):
        super(Attention, self).__init__()
        self.BN = keras.layers.BatchNormalization()
        self.GAP = keras.layers.GlobalAveragePooling2D()
        self.conv1 = keras.layers.Conv2D(filters, 1, 1)
        self.relu = keras.layers.ReLU()
        self.conv2 = keras.layers.Conv2D(filters, 1, 1)

        self.ASPP = ASPP(filters, size)

    def call(self, inputs):
        img_fts1 = self.ASPP(inputs)

        img_fts2 = self.GAP(self.BN(img_fts1))
        img_fts2 = tf.expand_dims(img_fts2, axis=1)
        img_fts2 = tf.expand_dims(img_fts2, axis=1)
        img_fts2 = self.conv2(self.relu(self.conv1(img_fts2)))
        img_fts2 = keras.activations.sigmoid(img_fts2)
        out = tf.einsum('mijn, mpqn -> mijn', img_fts1, img_fts2)
        return out


if __name__ == '__main__':
    img = tf.random.normal((2, 64, 64, 3))
    a = Attention(5, (64, 64))
    y = a(img)
    print(y.shape)
