import os

from tensorflow.keras import regularizers

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

'''
reference: 
    https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
paper:
    https://arxiv.org/pdf/1904.11492.pdf
'''

import config


class GlobalContext(keras.layers.Layer):
    def __init__(self, inplanes, ratio, pooling_type='avg', fusion_types=('channel_add', 'channel_mul'),
                 size=config.FTS_SIZE, channel=1024):
        super(GlobalContext, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.size = size
        self.channel = channel
        if pooling_type == 'att':
            self.conv_mask = keras.layers.Conv2D(
                filters=1,
                kernel_size=1,
                kernel_initializer=config.CONV_INIT,
            )
            self.softmax = keras.layers.Softmax(axis=-1)
        else:
            self.avg_pool = keras.layers.GlobalAveragePooling2D()
        if 'channel_add' in fusion_types:
            self.channel_add_conv = keras.Sequential([
                # keras.layers.LayerNormalization(
                #     axis=[1, 2, 3],
                #     epsilon=1e-5,
                #     trainable=True
                #     # beta_regularizer=config.L2,
                #     # gamma_regularizer=config.L2,
                # ),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    self.inplanes,
                    kernel_size=1,
                    kernel_initializer=config.CONV_INIT,
                    kernel_regularizer=config.L2,
                    # bias_regularizer=config.L2
                )
            ])
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = keras.Sequential([
                keras.layers.Conv2D(
                    self.planes,
                    kernel_size=1,
                    kernel_initializer=config.CONV_INIT,
                    kernel_regularizer=config.L2,
                    # bias_regularizer=config.L2
                ),
                # keras.layers.LayerNormalization(
                #     axis=[1, 2, 3],
                #     epsilon=1e-5,
                #     trainable=True,
                #     # beta_regularizer=config.L2,
                #     # gamma_regularizer=config.L2,
                # ),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    self.inplanes,
                    kernel_size=1,
                    kernel_initializer=config.CONV_INIT,
                    activation='sigmoid',
                    kernel_regularizer=config.L2,
                    # bias_regularizer=config.L2
                )
            ])
        else:
            self.channel_mul_conv = None

    def spatical_pool(self, x):
        if self.pooling_type == 'att':
            input_x = x
            input_x = tf.reshape(input_x, (-1, self.size * self.size, self.channel))
            input_x = tf.transpose(input_x, [0, 2, 1])
            input_x = tf.expand_dims(input_x, axis=1)

            context_mask = self.conv_mask(x)
            context_mask = tf.reshape(context_mask, (-1, self.size * self.size, 1))
            context_mask = tf.transpose(context_mask, [0, 2, 1])
            context_mask = self.softmax(context_mask)
            context_mask = tf.expand_dims(context_mask, axis=-1)
            context = tf.einsum('njcw, njwq -> njcq', input_x, context_mask)
            context = tf.transpose(context, [0, 1, 3, 2])
        else:
            context = self.avg_pool(x)
            context = tf.reshape(context, (-1, 1, 1, self.channel))
        return context

    def call(self, inputs, **kwargs):
        context = self.spatical_pool(inputs)
        if self.channel_mul_conv is not None:
            channel_mul_term = self.channel_mul_conv(context)
            out = inputs*channel_mul_term
        else:
            out = inputs
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class Route(keras.layers.Layer):
    def __init__(self, inplanes, ratio, channel):
        super(Route, self).__init__()
        self.conv = keras.layers.Conv2D(
            channel,
            1,
            1,
            kernel_initializer=config.CONV_INIT,
            kernel_regularizer=config.L2,
            # bias_regularizer=config.L2
        )
        self.globalContext = GlobalContext(inplanes=inplanes, ratio=ratio, channel=channel)
        self.globalAvgPooling = keras.layers.GlobalAveragePooling2D()
        self.l2Norm = tf.math.l2_normalize
        self.dense = keras.layers.Dense(
            1,
            kernel_initializer=config.DENSE_INIT,
            activation='sigmoid',
            kernel_regularizer=config.L2,
            # bias_regularizer=config.L2
        )

    def call(self, inputs, **kwargs):
        out = self.conv(inputs)
        out = self.globalContext(out)
        out = self.globalAvgPooling(out)
        out = tf.sign(out) * tf.math.sqrt(tf.sign(out) * out + 1e-12)
        out = self.l2Norm(out, axis=-1)
        out = self.dense(out)
        return out


if __name__ == '__main__':
    # test GlobalContext
    img = tf.random.normal((5, 28, 28, 512))
    gcLayer = GlobalContext(inplanes=512, ratio=1, channel=512)
    y = gcLayer(img)
    print(y.shape)

    # test Route
    # inputs = keras.Input(shape=(14, 14, 1024))
    # route = Route(inplanes=512, ratio=1, channel=512)
    # model = keras.Model(inputs=inputs, outputs=route(inputs))
    # y = model(img)
    # print(y)
